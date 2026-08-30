"""HPC bundle export, remote submission, and run monitoring workspace.

The widget deliberately keeps three concerns separate:

* :mod:`hpc_bundle` owns portable run-request creation and validation;
* :mod:`hpc_remote` owns credential-safe SSH/Plink process execution; and
* this module owns only user input, background dispatch, and tracked-run state.

Both services are injectable.  Apart from reading non-secret ``QSettings``, a
``RunsWorkspace`` constructor performs no file, network, or process operation.
That makes the tab safe to embed in GRIM and straightforward to exercise with
fake services in off-screen Qt tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Callable, Mapping
import uuid

from PySide6.QtCore import QObject, QSettings, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


_SETTINGS_PREFIX = "runs"
_TRACKED_RUNS_SCHEMA = "grim.runs.registry.v1"
_SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_WALLTIME = re.compile(r"^(?:\d+-)?\d{1,3}:[0-5]\d:[0-5]\d$")
_TERMINAL_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
    }
)
_RUN_TERMINAL_STATES = _TERMINAL_STATES | {"INCOMPLETE"}


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_slurm_state(value: Any) -> str:
    """Normalize Slurm state decorations before GUI state decisions."""

    state = str(value).strip().upper()
    if not state:
        return "UNKNOWN"
    state = state.split(None, 1)[0].rstrip("+")
    return {"OUT_OF_ME": "OUT_OF_MEMORY"}.get(state, state)


def _is_terminal_state(value: Any) -> bool:
    return _normalize_slurm_state(value) in _TERMINAL_STATES


def _is_run_terminal_state(value: Any) -> bool:
    """Include GRIM's local fail-closed terminal result state."""

    return _normalize_slurm_state(value) in _RUN_TERMINAL_STATES


def _plain_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    payload = getattr(value, "payload", None)
    if isinstance(payload, Mapping):
        return {str(key): item for key, item in payload.items()}
    return {}


def _read_member(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _parse_number_list(text: str, *, label: str) -> list[float]:
    """Parse comma-separated numbers and inclusive ``start:stop:step`` ranges."""

    raw = str(text).strip()
    if not raw:
        raise ValueError(f"{label} cannot be empty.")
    values: list[float] = []
    for token in (part.strip() for part in raw.split(",")):
        if not token:
            raise ValueError(f"{label} contains an empty item.")
        pieces = [piece.strip() for piece in token.split(":")]
        if len(pieces) == 1:
            try:
                value = float(pieces[0])
            except ValueError as exc:
                raise ValueError(f"{label} contains an invalid number: {token!r}.") from exc
            if not math.isfinite(value):
                raise ValueError(f"{label} values must be finite.")
            values.append(value)
            continue
        if len(pieces) != 3:
            raise ValueError(
                f"{label} range {token!r} must use start:stop:step."
            )
        try:
            start, stop, step = (float(piece) for piece in pieces)
        except ValueError as exc:
            raise ValueError(f"{label} contains an invalid range: {token!r}.") from exc
        if not all(math.isfinite(value) for value in (start, stop, step)):
            raise ValueError(f"{label} range values must be finite.")
        if step == 0.0 or (stop - start) * step < 0.0:
            raise ValueError(
                f"{label} range {token!r} has a step pointing away from its stop."
            )
        count = int(math.floor((stop - start) / step + 1.0e-10)) + 1
        if count < 1 or count > 100_000:
            raise ValueError(f"{label} range {token!r} has an unreasonable size.")
        values.extend(start + index * step for index in range(count))
    if not values:
        raise ValueError(f"{label} cannot be empty.")
    return [float(value) for value in values]


def _safe_remote_join(root: str, *parts: str) -> str:
    """Join already-validated remote path parts without Windows path rules."""

    base = str(root).strip()
    if not base.startswith("/"):
        raise ValueError("Remote workspace root must be an absolute Linux path.")
    if "\x00" in base or "\n" in base or "\r" in base:
        raise ValueError("Remote workspace root contains unsupported characters.")
    path = PurePosixPath(base)
    for part in parts:
        value = str(part).strip()
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"Unsafe remote path component: {part!r}.")
        path /= value
    return str(path)


@dataclass(frozen=True)
class ConnectionValues:
    """Non-secret connection metadata captured for one remote operation."""

    transport: str = "openssh"
    profile: str = ""
    host: str = ""
    username: str = ""
    port: int = 22
    identity_file: str = ""
    putty_host_key: str = ""
    python_executable: str = "python3"


@dataclass
class TrackedRun:
    """Small, non-secret local record used to reconnect to a SLURM run."""

    run_id: str
    solver: str
    bundle_id: str = ""
    job_ids: tuple[str, ...] = ()
    state: str = "STAGED"
    progress: str = ""
    remote_bundle: str = ""
    remote_cli: str = ""
    remote_stage_dir: str = ""
    remote_stage_result: str = ""
    remote_run_dir: str = ""
    remote_results: str = ""
    remote_log: str = ""
    local_bundle: str = ""
    results_complete: bool | None = None
    results_expected: int = 0
    results_present: int = 0
    updated_utc: str = field(default_factory=_utc_now_text)
    connection: dict[str, Any] = field(default_factory=dict)

    def to_json_value(self) -> dict[str, Any]:
        result = asdict(self)
        result["job_ids"] = list(self.job_ids)
        return result

    @classmethod
    def from_json_value(cls, value: Mapping[str, Any]) -> "TrackedRun":
        run_id = str(value.get("run_id", "")).strip()
        if not run_id or not _SAFE_RUN_NAME.fullmatch(run_id):
            raise ValueError("Tracked run has an invalid run_id.")
        raw_job_ids = value.get("job_ids", ())
        if isinstance(raw_job_ids, (str, bytes)):
            raw_job_ids = (raw_job_ids,)
        job_ids = tuple(
            str(item).strip() for item in raw_job_ids if str(item).strip()
        )
        raw_connection = dict(value.get("connection", {}) or {})
        connection = {
            name: raw_connection[name]
            for name in ConnectionValues.__dataclass_fields__
            if name in raw_connection
        }
        raw_complete = value.get("results_complete")
        results_complete = (
            raw_complete if type(raw_complete) is bool else None
        )

        def saved_count(name: str) -> int:
            candidate = value.get(name, 0)
            return candidate if type(candidate) is int and candidate >= 0 else 0

        return cls(
            run_id=run_id,
            solver=str(value.get("solver", "2d")),
            bundle_id=str(value.get("bundle_id", "")),
            job_ids=job_ids,
            state=str(value.get("state", "UNKNOWN")),
            progress=str(value.get("progress", "")),
            remote_bundle=str(value.get("remote_bundle", "")),
            remote_cli=str(value.get("remote_cli", "")),
            remote_stage_dir=str(value.get("remote_stage_dir", "")),
            remote_stage_result=str(value.get("remote_stage_result", "")),
            remote_run_dir=str(value.get("remote_run_dir", "")),
            remote_results=str(value.get("remote_results", "")),
            remote_log=str(value.get("remote_log", "")),
            local_bundle=str(value.get("local_bundle", "")),
            results_complete=results_complete,
            results_expected=saved_count("results_expected"),
            results_present=saved_count("results_present"),
            updated_utc=str(value.get("updated_utc", "")) or _utc_now_text(),
            connection=connection,
        )


class _OperationWorker(QObject):
    succeeded = Signal(object)
    failed = Signal(object)

    def __init__(self, operation: Callable[[], Any]) -> None:
        super().__init__()
        self._operation = operation

    @Slot()
    def run(self) -> None:
        try:
            result = self._operation()
        except Exception as exc:  # Boundary between backend and GUI.
            self.failed.emit(exc)
            return
        self.succeeded.emit(result)


class _DownloadVerificationError(RuntimeError):
    """The fresh pre-transfer manifest inventory was not authoritative."""

    def __init__(self, message: str, completion: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.completion = dict(completion or {})


class RunsWorkspace(QWidget):
    """Create portable HPC requests and manage explicit remote operations."""

    status_changed = Signal(str)
    run_submitted = Signal(object)
    results_downloaded = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        backend_path: str | os.PathLike[str] | None = None,
        bundle_service: Any | None = None,
        remote_client_factory: Callable[[Any], Any] | None = None,
        connection_config_factory: Callable[[ConnectionValues], Any] | None = None,
        settings: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self._backend_path = backend_path
        self._bundle_service_value = bundle_service
        self._remote_client_factory_value = remote_client_factory
        self._connection_config_factory_value = connection_config_factory
        self._settings = settings if settings is not None else QSettings("GRIM", "GRIM")
        self._thread: QThread | None = None
        self._worker: _OperationWorker | None = None
        self._active_kind = ""
        self._success_handler: Callable[[Any], None] | None = None
        self._failure_handler: Callable[[Exception], None] | None = None
        self._last_error = ""
        self._tracked_runs: dict[str, TrackedRun] = {}
        self._build_ui()
        self._load_settings()
        self._connect_signals()
        self._solver_changed()
        self._transport_changed()
        self._render_runs()

    # ------------------------------------------------------------------
    # Public shell contract
    # ------------------------------------------------------------------
    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def tracked_runs(self) -> tuple[TrackedRun, ...]:
        return tuple(self._tracked_runs.values())

    def job_is_running(self) -> bool:
        """Whether a foreground local transfer/command is active.

        Remote SLURM jobs intentionally do not count: they survive SSH and GRIM
        closing, so their presence must never trap the user in the application.
        """

        return bool(self._thread is not None and self._thread.isRunning())

    def busy_operation(self) -> str | None:
        if not self.job_is_running():
            return None
        return {
            "export": "HPC bundle export",
            "verify": "HPC bundle verification",
            "test": "HPC connection test",
            "submit": "HPC upload and submission",
            "refresh": "HPC status refresh",
            "cancel": "HPC cancellation request",
            "download": "HPC result download",
        }.get(self._active_kind, "HPC operation")

    def focus_workspace(self) -> None:
        self.btn_test_connection.setFocus(Qt.FocusReason.OtherFocusReason)

    def selected_run(self) -> TrackedRun | None:
        rows = self.jobs_table.selectionModel().selectedRows()
        if not rows:
            return None
        item = self.jobs_table.item(rows[0].row(), 0)
        if item is None:
            return None
        return self._tracked_runs.get(str(item.data(Qt.ItemDataRole.UserRole)))

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        heading = QLabel("HPC runs", self)
        heading.setObjectName("plotTitle")
        intro = QLabel(
            "Build the same portable request for manual transfer or upload and "
            "submit it to a headless Linux SLURM machine. GRIM closes each SSH "
            "connection after an operation; submitted jobs continue on the cluster.",
            self,
        )
        intro.setWordWrap(True)
        outer.addWidget(heading)
        outer.addWidget(intro)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)

        self.controls_scroll = QScrollArea(splitter)
        self.controls_scroll.setObjectName("runsControlsScroll")
        self.controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.controls_scroll.setMinimumWidth(430)
        self.controls_scroll.setMaximumWidth(550)
        self.controls_content = QWidget(self.controls_scroll)
        controls = QVBoxLayout(self.controls_content)
        controls.setContentsMargins(2, 2, 8, 2)
        controls.setSpacing(8)

        connection_group = QGroupBox("1  Connection", self.controls_content)
        connection_form = QFormLayout(connection_group)
        self.transport_combo = QComboBox(connection_group)
        self.transport_combo.addItem("Windows OpenSSH", "openssh")
        self.transport_combo.addItem("PuTTY saved session (Plink)", "putty")
        self.profile_edit = QLineEdit(connection_group)
        self.profile_edit.setPlaceholderText("Optional SSH config alias")
        self.host_edit = QLineEdit(connection_group)
        self.host_edit.setPlaceholderText("login.cluster.example")
        self.username_edit = QLineEdit(connection_group)
        self.username_edit.setPlaceholderText("Linux username")
        self.port_spin = QSpinBox(connection_group)
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(22)
        self.identity_edit = QLineEdit(connection_group)
        self.identity_edit.setPlaceholderText("Optional private-key path")
        self.btn_identity = QPushButton("Choose…", connection_group)
        identity_row = QWidget(connection_group)
        identity_layout = QHBoxLayout(identity_row)
        identity_layout.setContentsMargins(0, 0, 0, 0)
        identity_layout.addWidget(self.identity_edit, 1)
        identity_layout.addWidget(self.btn_identity)
        self.putty_host_key_edit = QLineEdit(connection_group)
        self.putty_host_key_edit.setPlaceholderText(
            "Optional verified fingerprint; cached PuTTY key otherwise"
        )
        self.connection_hint = QLabel(connection_group)
        self.connection_hint.setWordWrap(True)
        connection_form.addRow("Transport", self.transport_combo)
        connection_form.addRow("Profile / session", self.profile_edit)
        connection_form.addRow("Host", self.host_edit)
        connection_form.addRow("Username", self.username_edit)
        connection_form.addRow("SSH port", self.port_spin)
        connection_form.addRow("Identity file", identity_row)
        connection_form.addRow("PuTTY host key", self.putty_host_key_edit)
        connection_form.addRow(self.connection_hint)
        self.btn_test_connection = QPushButton("Test Connection", connection_group)
        connection_form.addRow(self.btn_test_connection)
        controls.addWidget(connection_group)

        request_group = QGroupBox("2  Portable run request", self.controls_content)
        request_layout = QVBoxLayout(request_group)
        request_form = QFormLayout()
        self.solver_combo = QComboBox(request_group)
        self.solver_combo.addItem("2-D arbitrary geometry", "2d")
        self.solver_combo.addItem("Body of revolution (BoR)", "bor")
        self.frequency_edit = QLineEdit("2, 4, 6, 8, 10", request_group)
        self.frequency_edit.setToolTip("Comma list or inclusive start:stop:step range")
        self.azimuth_edit = QLineEdit("0:180:30", request_group)
        self.azimuth_edit.setToolTip("Comma list or inclusive start:stop:step range")
        self.elevation_edit = QLineEdit("-60:60:5", request_group)
        self.elevation_label = QLabel("Elevations (deg)", request_group)
        self.body_axis_az_spin = QDoubleSpinBox(request_group)
        self.body_axis_az_spin.setRange(-360.0, 360.0)
        self.body_axis_az_spin.setDecimals(3)
        self.body_axis_az_spin.setSuffix("°")
        self.body_axis_el_spin = QDoubleSpinBox(request_group)
        self.body_axis_el_spin.setRange(-180.0, 180.0)
        self.body_axis_el_spin.setDecimals(3)
        self.body_axis_el_spin.setSuffix("°")
        self.body_roll_spin = QDoubleSpinBox(request_group)
        self.body_roll_spin.setRange(-360.0, 360.0)
        self.body_roll_spin.setDecimals(3)
        self.body_roll_spin.setSuffix("°")
        self.body_axis_az_label = QLabel("Body-axis azimuth", request_group)
        self.body_axis_el_label = QLabel("Body-axis elevation", request_group)
        self.body_roll_label = QLabel("Body roll", request_group)
        self.units_combo = QComboBox(request_group)
        self.units_combo.addItem("inches", "inches")
        self.units_combo.addItem("meters", "meters")
        request_form.addRow("Solver", self.solver_combo)
        request_form.addRow("Frequencies (GHz)", self.frequency_edit)
        request_form.addRow("Azimuths (deg)", self.azimuth_edit)
        request_form.addRow(self.elevation_label, self.elevation_edit)
        request_form.addRow(self.body_axis_az_label, self.body_axis_az_spin)
        request_form.addRow(self.body_axis_el_label, self.body_axis_el_spin)
        request_form.addRow(self.body_roll_label, self.body_roll_spin)
        request_form.addRow("Geometry units", self.units_combo)
        request_layout.addLayout(request_form)

        geometry_help = QLabel(
            "Add saved .geo files and label their physical role. The bundle "
            "copies required material/IBC sidecars with each geometry.",
            request_group,
        )
        geometry_help.setWordWrap(True)
        request_layout.addWidget(geometry_help)
        self.geometry_list = QListWidget(request_group)
        self.geometry_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.geometry_list.setMinimumHeight(100)
        request_layout.addWidget(self.geometry_list)
        geometry_buttons = QHBoxLayout()
        self.btn_add_frd = QPushButton("Add FRD…", request_group)
        self.btn_add_opn = QPushButton("Add OPN…", request_group)
        self.btn_add_bor = QPushButton("Add BoR…", request_group)
        self.btn_remove_geometry = QPushButton("Remove", request_group)
        geometry_buttons.addWidget(self.btn_add_frd)
        geometry_buttons.addWidget(self.btn_add_opn)
        geometry_buttons.addWidget(self.btn_add_bor)
        geometry_buttons.addWidget(self.btn_remove_geometry)
        request_layout.addLayout(geometry_buttons)

        schedule_form = QFormLayout()
        self.nodes_spin = QSpinBox(request_group)
        self.nodes_spin.setRange(1, 10_000)
        self.nodes_spin.setValue(1)
        self.jobs_spin = QSpinBox(request_group)
        self.jobs_spin.setRange(1, 1_000)
        self.jobs_spin.setValue(1)
        self.partition_edit = QLineEdit("compute", request_group)
        self.account_edit = QLineEdit(request_group)
        self.qos_edit = QLineEdit(request_group)
        self.walltime_edit = QLineEdit(request_group)
        self.walltime_edit.setPlaceholderText("Optional HH:MM:SS")
        self.cores_spin = QSpinBox(request_group)
        self.cores_spin.setRange(0, 4096)
        self.cores_spin.setSpecialValueText("Whole node")
        self.memory_edit = QLineEdit("0", request_group)
        self.memory_edit.setToolTip("SLURM memory value, e.g. 0 for all or 64G")
        self.mesh_certification_check = QCheckBox(
            "Require base/fine mesh certification", request_group
        )
        self.mesh_certification_check.setChecked(True)
        schedule_form.addRow("Nodes per job", self.nodes_spin)
        schedule_form.addRow("Job arrays", self.jobs_spin)
        schedule_form.addRow("Partition", self.partition_edit)
        schedule_form.addRow("Account", self.account_edit)
        schedule_form.addRow("QOS", self.qos_edit)
        schedule_form.addRow("Walltime", self.walltime_edit)
        schedule_form.addRow("Cores per node", self.cores_spin)
        schedule_form.addRow("Memory per node", self.memory_edit)
        schedule_form.addRow(self.mesh_certification_check)
        request_layout.addLayout(schedule_form)

        bundle_form = QFormLayout()
        self.bundle_path_edit = QLineEdit(request_group)
        self.bundle_path_edit.setPlaceholderText("New or existing portable bundle folder")
        self.btn_bundle_path = QPushButton("Choose…", request_group)
        bundle_row = QWidget(request_group)
        bundle_row_layout = QHBoxLayout(bundle_row)
        bundle_row_layout.setContentsMargins(0, 0, 0, 0)
        bundle_row_layout.addWidget(self.bundle_path_edit, 1)
        bundle_row_layout.addWidget(self.btn_bundle_path)
        bundle_form.addRow("Bundle folder", bundle_row)
        request_layout.addLayout(bundle_form)
        bundle_buttons = QHBoxLayout()
        self.btn_export_bundle = QPushButton("Export Bundle", request_group)
        self.btn_verify_bundle = QPushButton("Verify Existing", request_group)
        bundle_buttons.addWidget(self.btn_export_bundle)
        bundle_buttons.addWidget(self.btn_verify_bundle)
        request_layout.addLayout(bundle_buttons)
        export_note = QLabel(
            "Export Bundle is the manual-transfer path: copy that folder to "
            "Linux and run the command recorded in its README.",
            request_group,
        )
        export_note.setWordWrap(True)
        request_layout.addWidget(export_note)
        controls.addWidget(request_group)

        submit_group = QGroupBox("3  Upload and submit", self.controls_content)
        submit_form = QFormLayout(submit_group)
        default_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_name_edit = QLineEdit(default_name, submit_group)
        self.remote_root_edit = QLineEdit(submit_group)
        self.remote_root_edit.setPlaceholderText("/home/user/grim_hpc")
        self.remote_cli_edit = QLineEdit(submit_group)
        self.remote_cli_edit.setPlaceholderText(
            "/home/user/GHOST/Backend/hpc_bundle.py"
        )
        self.remote_python_edit = QLineEdit("python3", submit_group)
        self.remote_python_edit.setPlaceholderText("python3 or /path/to/venv/bin/python")
        self.remote_python_edit.setToolTip(
            "Python on the Linux login node. Use a virtual-environment Python "
            "when the cluster's default python3 lacks GHOST dependencies."
        )
        submit_form.addRow("Run name", self.run_name_edit)
        submit_form.addRow("Remote workspace root", self.remote_root_edit)
        submit_form.addRow("Remote hpc_bundle.py", self.remote_cli_edit)
        submit_form.addRow("Remote Python", self.remote_python_edit)
        self.btn_upload_submit = QPushButton("Upload & Submit", submit_group)
        self.btn_upload_submit.setToolTip(
            "Build a fresh temporary bundle from the form above, upload it, "
            "stage it on Linux, and submit its generated SLURM scripts."
        )
        submit_form.addRow(self.btn_upload_submit)
        submit_note = QLabel(
            "No password is stored or passed on the command line. Use an "
            "approved SSH agent/key or a PuTTY saved session, and verify the "
            "server host key before the first GRIM connection.",
            submit_group,
        )
        submit_note.setWordWrap(True)
        submit_form.addRow(submit_note)
        controls.addWidget(submit_group)
        controls.addStretch(1)
        self.controls_scroll.setWidget(self.controls_content)

        right = QWidget(splitter)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 2, 2, 2)
        jobs_header = QHBoxLayout()
        jobs_title = QLabel("Tracked SLURM runs", right)
        jobs_title.setObjectName("plotTitle")
        jobs_header.addWidget(jobs_title)
        jobs_header.addStretch(1)
        self.btn_refresh = QPushButton("Refresh", right)
        self.btn_cancel = QPushButton("Cancel Job", right)
        self.btn_download = QPushButton("Download Results…", right)
        jobs_header.addWidget(self.btn_refresh)
        jobs_header.addWidget(self.btn_cancel)
        jobs_header.addWidget(self.btn_download)
        right_layout.addLayout(jobs_header)

        self.jobs_table = QTableWidget(0, 7, right)
        self.jobs_table.setHorizontalHeaderLabels(
            ["Run", "Solver", "Job ID(s)", "State", "Progress", "Remote folder", "Updated"]
        )
        self.jobs_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.jobs_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.jobs_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.jobs_table.horizontalHeader().setStretchLastSection(True)
        self.jobs_table.setMinimumHeight(190)
        right_layout.addWidget(self.jobs_table, 1)

        log_title = QLabel("Connection and scheduler log", right)
        log_title.setObjectName("plotTitle")
        right_layout.addWidget(log_title)
        self.log_view = QPlainTextEdit(right)
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log_view.setMinimumHeight(170)
        right_layout.addWidget(self.log_view, 1)
        self.status_label = QLabel("Ready.", right)
        self.status_label.setWordWrap(True)
        right_layout.addWidget(self.status_label)

        splitter.addWidget(self.controls_scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([500, 900])

    def _connect_signals(self) -> None:
        self.transport_combo.currentIndexChanged.connect(self._transport_changed)
        self.profile_edit.textChanged.connect(self._profile_changed)
        self.solver_combo.currentIndexChanged.connect(self._solver_changed)
        self.btn_identity.clicked.connect(self._choose_identity)
        self.btn_test_connection.clicked.connect(self.test_connection)
        self.btn_add_frd.clicked.connect(lambda: self._choose_geometries("FRD"))
        self.btn_add_opn.clicked.connect(lambda: self._choose_geometries("OPN"))
        self.btn_add_bor.clicked.connect(lambda: self._choose_geometries("BOR"))
        self.btn_remove_geometry.clicked.connect(self._remove_selected_geometries)
        self.btn_bundle_path.clicked.connect(self._choose_bundle)
        self.btn_export_bundle.clicked.connect(self.export_bundle)
        self.btn_verify_bundle.clicked.connect(self.verify_bundle)
        self.btn_upload_submit.clicked.connect(self.upload_and_submit)
        self.btn_refresh.clicked.connect(self.refresh_selected_run)
        self.btn_cancel.clicked.connect(self.cancel_selected_run)
        self.btn_download.clicked.connect(self.download_selected_results)
        self.jobs_table.itemSelectionChanged.connect(self._run_selection_changed)

    # ------------------------------------------------------------------
    # Settings and state
    # ------------------------------------------------------------------
    def _setting(self, name: str, default: Any = "") -> Any:
        return self._settings.value(f"{_SETTINGS_PREFIX}/{name}", default)

    def _load_settings(self) -> None:
        transport = str(self._setting("transport", "openssh"))
        index = self.transport_combo.findData(transport)
        if index >= 0:
            self.transport_combo.setCurrentIndex(index)
        self.profile_edit.setText(str(self._setting("profile", "")))
        self.host_edit.setText(str(self._setting("host", "")))
        self.username_edit.setText(str(self._setting("username", "")))
        try:
            self.port_spin.setValue(int(self._setting("port", 22)))
        except (TypeError, ValueError):
            self.port_spin.setValue(22)
        self.identity_edit.setText(str(self._setting("identity_file", "")))
        self.putty_host_key_edit.setText(str(self._setting("putty_host_key", "")))
        self.bundle_path_edit.setText(str(self._setting("bundle_path", "")))
        self.remote_root_edit.setText(str(self._setting("remote_root", "")))
        self.remote_cli_edit.setText(str(self._setting("remote_cli", "")))
        self.remote_python_edit.setText(
            str(self._setting("remote_python", "python3")) or "python3"
        )

        solver = str(self._setting("solver", "2d"))
        solver_index = self.solver_combo.findData(solver)
        if solver_index >= 0:
            self.solver_combo.setCurrentIndex(solver_index)
        self.frequency_edit.setText(
            str(self._setting("frequencies", self.frequency_edit.text()))
        )
        self.azimuth_edit.setText(
            str(self._setting("azimuths", self.azimuth_edit.text()))
        )
        self.elevation_edit.setText(
            str(self._setting("elevations", self.elevation_edit.text()))
        )
        units = str(self._setting("geometry_units", "inches"))
        units_index = self.units_combo.findData(units)
        if units_index >= 0:
            self.units_combo.setCurrentIndex(units_index)
        for widget, key, default in (
            (self.nodes_spin, "nodes", 1),
            (self.jobs_spin, "jobs", 1),
            (self.cores_spin, "cores", 0),
        ):
            try:
                widget.setValue(int(self._setting(key, default)))
            except (TypeError, ValueError):
                widget.setValue(default)
        self.partition_edit.setText(str(self._setting("partition", "compute")))
        self.account_edit.setText(str(self._setting("account", "")))
        self.qos_edit.setText(str(self._setting("qos", "")))
        self.walltime_edit.setText(str(self._setting("walltime", "")))
        self.memory_edit.setText(str(self._setting("memory", "0")))
        mesh_raw = self._setting("mesh_certification", True)
        self.mesh_certification_check.setChecked(
            mesh_raw
            if isinstance(mesh_raw, bool)
            else str(mesh_raw).strip().casefold() in {"1", "true", "yes", "on"}
        )
        for widget, key in (
            (self.body_axis_az_spin, "body_axis_az"),
            (self.body_axis_el_spin, "body_axis_el"),
            (self.body_roll_spin, "body_roll"),
        ):
            try:
                widget.setValue(float(self._setting(key, 0.0)))
            except (TypeError, ValueError):
                widget.setValue(0.0)
        raw_geometries = self._setting("geometries", "")
        if raw_geometries:
            try:
                entries = json.loads(str(raw_geometries))
                for entry in entries:
                    try:
                        self.add_geometry(entry["role"], entry["path"])
                    except (KeyError, OSError, TypeError, ValueError):
                        continue
            except (TypeError, ValueError, json.JSONDecodeError):
                pass

        raw_runs = self._setting("tracked_runs", "")
        if raw_runs:
            try:
                document = json.loads(str(raw_runs))
                if not isinstance(document, Mapping):
                    raise ValueError("registry is not an object")
                if document.get("schema") != _TRACKED_RUNS_SCHEMA:
                    raise ValueError("unsupported schema")
                values = document.get("runs", ())
                if isinstance(values, (str, bytes)) or not isinstance(values, list):
                    raise ValueError("runs is not a list")
            except (TypeError, ValueError, json.JSONDecodeError):
                # A corrupt convenience registry must not prevent GRIM launch.
                self._tracked_runs.clear()
                self._append_log("WARNING: Ignored an unreadable tracked-run registry.")
            else:
                skipped = 0
                for value in values:
                    try:
                        if not isinstance(value, Mapping):
                            raise ValueError("tracked run is not an object")
                        run = TrackedRun.from_json_value(value)
                    except (TypeError, ValueError):
                        skipped += 1
                        continue
                    self._tracked_runs[run.run_id] = run
                if skipped:
                    self._append_log(
                        f"WARNING: Kept valid tracked runs and skipped {skipped} "
                        "corrupt registry entr" + ("y." if skipped == 1 else "ies.")
                    )

    def save_settings(self) -> bool:
        """Persist non-secret preferences and the recovery registry.

        A tracked run is useful only if it survives a process interruption, so
        callers that are about to begin remote work can require a durable local
        journal before uploading or submitting anything.
        """
        values = self._connection_values()
        for name, value in (
            ("transport", values.transport),
            ("profile", values.profile),
            ("host", values.host),
            ("username", values.username),
            ("port", values.port),
            ("identity_file", values.identity_file),
            ("putty_host_key", values.putty_host_key),
            ("bundle_path", self.bundle_path_edit.text().strip()),
            ("remote_root", self.remote_root_edit.text().strip()),
            ("remote_cli", self.remote_cli_edit.text().strip()),
            ("remote_python", self.remote_python_edit.text().strip() or "python3"),
            ("solver", str(self.solver_combo.currentData())),
            ("frequencies", self.frequency_edit.text().strip()),
            ("azimuths", self.azimuth_edit.text().strip()),
            ("elevations", self.elevation_edit.text().strip()),
            ("geometry_units", str(self.units_combo.currentData())),
            ("nodes", int(self.nodes_spin.value())),
            ("jobs", int(self.jobs_spin.value())),
            ("partition", self.partition_edit.text().strip()),
            ("account", self.account_edit.text().strip()),
            ("qos", self.qos_edit.text().strip()),
            ("walltime", self.walltime_edit.text().strip()),
            ("cores", int(self.cores_spin.value())),
            ("memory", self.memory_edit.text().strip()),
            ("mesh_certification", self.mesh_certification_check.isChecked()),
            ("body_axis_az", float(self.body_axis_az_spin.value())),
            ("body_axis_el", float(self.body_axis_el_spin.value())),
            ("body_roll", float(self.body_roll_spin.value())),
            ("geometries", json.dumps(self.geometries(), separators=(",", ":"))),
        ):
            self._settings.setValue(f"{_SETTINGS_PREFIX}/{name}", value)
        registry = {
            "schema": _TRACKED_RUNS_SCHEMA,
            "runs": [run.to_json_value() for run in self._tracked_runs.values()],
        }
        self._settings.setValue(
            f"{_SETTINGS_PREFIX}/tracked_runs",
            json.dumps(registry, sort_keys=True, separators=(",", ":")),
        )
        sync = getattr(self._settings, "sync", None)
        if callable(sync):
            sync()
        status = getattr(self._settings, "status", None)
        if callable(status) and status() != QSettings.Status.NoError:
            self._append_log(
                "ERROR: Runs settings could not be written; the recovery registry "
                "is not durable."
            )
            return False
        return True

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API name
        if self.job_is_running():
            self._set_status(
                f"{self.busy_operation()} is still running; wait for it before closing."
            )
            event.ignore()
            return
        self.save_settings()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Form snapshots and validation
    # ------------------------------------------------------------------
    def _connection_values(self) -> ConnectionValues:
        return ConnectionValues(
            transport=str(self.transport_combo.currentData()),
            profile=self.profile_edit.text().strip(),
            host=self.host_edit.text().strip(),
            username=self.username_edit.text().strip(),
            port=int(self.port_spin.value()),
            identity_file=self.identity_edit.text().strip(),
            putty_host_key=self.putty_host_key_edit.text().strip(),
            python_executable=self.remote_python_edit.text().strip() or "python3",
        )

    def _validate_connection_values(self, values: ConnectionValues) -> None:
        if values.transport == "openssh":
            if not values.profile and not values.host:
                raise ValueError("Enter an OpenSSH config alias or an HPC hostname.")
            if values.identity_file and not Path(values.identity_file).expanduser().is_file():
                raise ValueError(f"SSH identity file not found: {values.identity_file}")
            return
        if values.transport == "putty":
            if not values.profile:
                raise ValueError("Enter the saved PuTTY session name used for this HPC.")
            return
        raise ValueError(f"Unsupported remote transport: {values.transport!r}.")

    def _connection_config(self, values: ConnectionValues) -> Any:
        self._validate_connection_values(values)
        if self._connection_config_factory_value is not None:
            return self._connection_config_factory_value(values)
        from hpc_remote import ConnectionConfig

        if values.transport == "putty":
            kwargs: dict[str, Any] = {}
            if values.putty_host_key:
                kwargs["host_key"] = values.putty_host_key
            return ConnectionConfig.putty_saved_session(values.profile, **kwargs)
        if values.profile:
            return ConnectionConfig.openssh_alias(values.profile)
        kwargs = {"port": values.port}
        if values.username:
            kwargs["username"] = values.username
        if values.identity_file:
            kwargs["identity_file"] = values.identity_file
        return ConnectionConfig.openssh_host(values.host, **kwargs)

    def _remote_client(self, values: ConnectionValues) -> Any:
        config = self._connection_config(values)
        if self._remote_client_factory_value is not None:
            return self._remote_client_factory_value(config)
        from hpc_remote import HpcRemoteClient

        return HpcRemoteClient(config)

    def _bundle_service(self) -> Any:
        if self._bundle_service_value is None:
            from ghost_integration import load_ghost_module

            self._bundle_service_value = load_ghost_module(
                "hpc_bundle", self._backend_path
            )
        return self._bundle_service_value

    def add_geometry(self, role: str, path: str | os.PathLike[str]) -> bool:
        normalized_role = str(role).strip().upper()
        if normalized_role not in {"FRD", "OPN", "BOR"}:
            raise ValueError(f"Unknown geometry role: {role!r}.")
        source = Path(path).expanduser().resolve()
        if source.suffix.lower() != ".geo" or not source.is_file():
            raise ValueError(f"Geometry must be a readable .geo file: {source}")
        for index in range(self.geometry_list.count()):
            value = self.geometry_list.item(index).data(Qt.ItemDataRole.UserRole)
            if isinstance(value, Mapping) and Path(str(value.get("path"))).resolve() == source:
                return False
        item = QListWidgetItem(f"[{normalized_role}]  {source.name}   —   {source.parent}")
        item.setData(
            Qt.ItemDataRole.UserRole,
            {"role": normalized_role, "path": str(source)},
        )
        item.setToolTip(str(source))
        self.geometry_list.addItem(item)
        return True

    def geometries(self) -> list[dict[str, str]]:
        return [
            dict(self.geometry_list.item(index).data(Qt.ItemDataRole.UserRole))
            for index in range(self.geometry_list.count())
        ]

    def _request_snapshot(self, *, require_bundle_path: bool = True) -> dict[str, Any]:
        solver = str(self.solver_combo.currentData())
        geometries = self.geometries()
        if not geometries:
            raise ValueError("Add at least one saved .geo geometry.")
        allowed_roles = {"FRD", "OPN"} if solver == "2d" else {"BOR"}
        incompatible = sorted(
            {entry["role"] for entry in geometries} - allowed_roles
        )
        if incompatible:
            raise ValueError(
                f"{solver.upper()} cannot use geometry role(s): "
                + ", ".join(incompatible)
                + ". Remove them or choose the matching solver."
            )
        stems = [Path(entry["path"]).stem.casefold() for entry in geometries]
        if len(stems) != len(set(stems)):
            raise ValueError("Geometry filename stems must be unique in one HPC bundle.")

        settings: dict[str, Any] = {
            "FREQUENCIES_GHZ": _parse_number_list(
                self.frequency_edit.text(), label="Frequencies"
            ),
            "AZIMUTHS_DEG": _parse_number_list(
                self.azimuth_edit.text(), label="Azimuths"
            ),
            "GEOMETRY_UNITS": str(self.units_combo.currentData()),
            "N_NODES": int(self.nodes_spin.value()),
            "N_JOBS": int(self.jobs_spin.value()),
            "SLURM_PARTITION": self.partition_edit.text().strip(),
            "MEM_PER_NODE": self.memory_edit.text().strip(),
            "MESH_CERTIFICATION": self.mesh_certification_check.isChecked(),
        }
        if not settings["SLURM_PARTITION"]:
            raise ValueError("SLURM partition cannot be empty.")
        if not settings["MEM_PER_NODE"]:
            raise ValueError("Memory per node cannot be empty.")
        if solver == "bor":
            settings["ELEVATIONS_DEG"] = _parse_number_list(
                self.elevation_edit.text(), label="Elevations"
            )
            settings["BODY_AXIS_AZ_DEG"] = float(self.body_axis_az_spin.value())
            settings["BODY_AXIS_EL_DEG"] = float(self.body_axis_el_spin.value())
            settings["BODY_ROLL_DEG"] = float(self.body_roll_spin.value())
        optional = (
            ("SLURM_ACCOUNT", self.account_edit.text().strip()),
            ("SLURM_QOS", self.qos_edit.text().strip()),
            ("SLURM_TIME", self.walltime_edit.text().strip()),
        )
        for key, value in optional:
            if value:
                settings[key] = value
        if settings.get("SLURM_TIME") and not _WALLTIME.fullmatch(
            str(settings["SLURM_TIME"])
        ):
            raise ValueError("Walltime must use HH:MM:SS or D-HH:MM:SS.")
        if self.cores_spin.value() > 0:
            settings["CORES_PER_NODE"] = int(self.cores_spin.value())

        raw_bundle_path = self.bundle_path_edit.text().strip()
        if require_bundle_path and not raw_bundle_path:
            raise ValueError("Choose a portable bundle folder.")
        bundle_path = (
            str(Path(raw_bundle_path).expanduser().resolve())
            if raw_bundle_path
            else ""
        )
        if require_bundle_path and Path(bundle_path).exists():
            raise ValueError(
                "Export Bundle publishes only to a new folder so existing files "
                "cannot be overwritten. Choose a new bundle folder, or use "
                "Verify Existing for this one."
            )
        return {
            "bundle_path": bundle_path,
            "solver": solver,
            "geometries": geometries,
            "settings": settings,
            # Bundle identity is a content/provenance UUID generated by the
            # bundle service.  The separate run-name field remains a human
            # label and remote-directory component.
            "bundle_id": None,
        }

    def _submission_snapshot(self) -> dict[str, Any]:
        request = self._request_snapshot(require_bundle_path=False)
        run_name = self.run_name_edit.text().strip()
        if not _SAFE_RUN_NAME.fullmatch(run_name):
            raise ValueError(
                "Run name must start with a letter or digit and use at most 80 "
                "letters, digits, dots, underscores, or hyphens."
            )
        if run_name in self._tracked_runs:
            raise ValueError(
                f"Run name {run_name!r} is already tracked. Choose a unique "
                "name so an earlier SLURM job is not hidden or overwritten."
            )
        remote_root = self.remote_root_edit.text().strip()
        remote_cli = self.remote_cli_edit.text().strip()
        if not remote_cli.startswith("/") or any(
            marker in remote_cli for marker in ("\x00", "\n", "\r")
        ):
            raise ValueError("Remote hpc_bundle.py must be an absolute Linux path.")
        python_executable = self.remote_python_edit.text().strip()
        if (
            not python_executable
            or python_executable.startswith("-")
            or any(ord(character) < 32 or ord(character) == 127 for character in python_executable)
        ):
            raise ValueError(
                "Remote Python must be a command such as python3 or an absolute "
                "virtual-environment interpreter path."
            )
        bundle_id = uuid.uuid4().hex
        request["bundle_id"] = bundle_id
        remote_upload_parent = _safe_remote_join(remote_root, "incoming", run_name)
        remote_bundle = _safe_remote_join(
            remote_upload_parent, f"bundle_{bundle_id}"
        )
        remote_workspace = _safe_remote_join(remote_root, "workspaces")
        remote_stage_dir = _safe_remote_join(
            remote_workspace, f"grim_{bundle_id}"
        )
        remote_stage_result = _safe_remote_join(
            remote_stage_dir, "stage_result.json"
        )
        connection = self._connection_values()
        self._validate_connection_values(connection)
        return {
            "request": request,
            "run_name": run_name,
            "bundle_id": bundle_id,
            "remote_root": remote_root,
            "remote_cli": remote_cli,
            "python_executable": python_executable,
            "remote_upload_parent": remote_upload_parent,
            "remote_bundle": remote_bundle,
            "remote_workspace": remote_workspace,
            "remote_stage_dir": remote_stage_dir,
            "remote_stage_result": remote_stage_result,
            "connection": connection,
            "upload_started": False,
            "stage_started": False,
        }

    # ------------------------------------------------------------------
    # User actions
    # ------------------------------------------------------------------
    @Slot()
    def test_connection(self) -> bool:
        try:
            values = self._connection_values()
            self._validate_connection_values(values)
        except Exception as exc:
            self._show_error(str(exc))
            return False

        def operation() -> Any:
            return self._remote_client(values).test_connection()

        return self._start_operation("test", operation, self._connection_succeeded)

    @Slot()
    def export_bundle(self) -> bool:
        try:
            request = self._request_snapshot()
        except Exception as exc:
            self._show_error(str(exc))
            return False

        def operation() -> Any:
            service = self._bundle_service()
            return service.create_portable_bundle(
                request["bundle_path"],
                solver=request["solver"],
                geometries=request["geometries"],
                settings=request["settings"],
                bundle_id=request["bundle_id"],
            )

        return self._start_operation(
            "export",
            operation,
            lambda result: self._bundle_exported(request["bundle_path"], result),
        )

    @Slot()
    def verify_bundle(self) -> bool:
        path = Path(self.bundle_path_edit.text().strip()).expanduser().resolve()
        if not path.is_dir():
            self._show_error(f"Portable bundle folder not found: {path}")
            return False

        def operation() -> Any:
            return self._bundle_service().verify_portable_bundle(path)

        return self._start_operation(
            "verify", operation, lambda result: self._bundle_verified(path, result)
        )

    @Slot()
    def upload_and_submit(self) -> bool:
        if self.job_is_running():
            self._show_error(f"{self.busy_operation()} is already running.")
            return False
        try:
            snapshot = self._submission_snapshot()
        except Exception as exc:
            self._show_error(str(exc))
            return False

        tracked = TrackedRun(
            run_id=str(snapshot["run_name"]),
            solver=str(snapshot["request"]["solver"]),
            bundle_id=str(snapshot["bundle_id"]),
            state="UPLOADING",
            progress="Building and uploading portable request",
            remote_bundle=str(snapshot["remote_bundle"]),
            remote_cli=str(snapshot["remote_cli"]),
            remote_stage_dir=str(snapshot["remote_stage_dir"]),
            remote_stage_result=str(snapshot["remote_stage_result"]),
            connection=asdict(snapshot["connection"]),
        )
        self._tracked_runs[tracked.run_id] = tracked
        self._render_runs(select_run_id=tracked.run_id)
        if not self.save_settings():
            self._tracked_runs.pop(tracked.run_id, None)
            self._render_runs()
            self._show_error(
                "The local run recovery record could not be saved, so no remote "
                "upload or submission was started. Check settings-file permissions."
            )
            return False

        def operation() -> Any:
            service = self._bundle_service()
            request_values = snapshot["request"]
            with tempfile.TemporaryDirectory(prefix="grim-hpc-upload-") as temporary:
                local_bundle = Path(temporary) / f"bundle_{snapshot['bundle_id']}"
                service.create_portable_bundle(
                    local_bundle,
                    solver=request_values["solver"],
                    geometries=request_values["geometries"],
                    settings=request_values["settings"],
                    bundle_id=request_values["bundle_id"],
                )
                request = service.verify_portable_bundle(local_bundle)
                client = self._remote_client(snapshot["connection"])
                snapshot["upload_started"] = True
                transfer = client.upload_bundle(
                    local_bundle, snapshot["remote_upload_parent"]
                )
                uploaded_remote = str(
                    _read_member(transfer, "remote_path", "")
                    or snapshot["remote_bundle"]
                )
                stage = getattr(client, "stage_hpc_bundle", None)
                snapshot["stage_started"] = True
                if callable(stage):
                    result = stage(
                        snapshot["remote_cli"],
                        uploaded_remote,
                        snapshot["remote_workspace"],
                        run_driver=True,
                        submit=True,
                        python_executable=snapshot["python_executable"],
                    )
                else:
                    result = client.invoke_hpc_bundle(
                        snapshot["remote_cli"],
                        (
                            "stage",
                            uploaded_remote,
                            "--workspace-root",
                            snapshot["remote_workspace"],
                            "--run-driver",
                            "--submit",
                        ),
                        python_executable=snapshot["python_executable"],
                    )
                return {"request": request, "transfer": transfer, "result": result}

        started = self._start_operation(
            "submit",
            operation,
            lambda result: self._submission_succeeded(snapshot, result),
            failure_handler=lambda exc: self._submission_failed(snapshot, exc),
        )
        if not started:
            self._tracked_runs.pop(tracked.run_id, None)
            self._render_runs()
            self.save_settings()
        return started

    @Slot()
    def refresh_selected_run(self) -> bool:
        run = self.selected_run()
        if run is None:
            self._show_error("Select one tracked run to refresh.")
            return False
        values = ConnectionValues(**run.connection)

        def operation() -> Any:
            client = self._remote_client(values)
            stage_result = None
            job_ids = tuple(run.job_ids)
            remote_run_dir = run.remote_run_dir
            remote_log = run.remote_log
            recovery_states = {
                "UPLOADING",
                "STAGING",
                "SUBMISSION UNKNOWN",
                "PARTIAL SUBMISSION",
                "FAILED",
            }
            if run.remote_stage_result and (
                not job_ids or run.state.upper() in recovery_states
            ):
                recover = getattr(client, "recover_hpc_bundle", None)
                reader = getattr(client, "read_stage_result", None)
                if callable(recover) and run.remote_cli and run.remote_stage_dir:
                    stage_result = recover(
                        run.remote_cli,
                        run.remote_stage_dir,
                        python_executable=values.python_executable,
                    )
                elif callable(reader):
                    stage_result = reader(run.remote_stage_result)
                if stage_result is not None:
                    payload = _plain_mapping(stage_result)
                    raw_ids = _read_member(stage_result, "job_ids", ()) or payload.get(
                        "job_ids", ()
                    )
                    job_ids = tuple(
                        str(item).strip() for item in raw_ids if str(item).strip()
                    )
                    remote_run_dir = str(
                        payload.get("run_dir") or remote_run_dir or ""
                    )
                    remote_log = str(payload.get("log_path") or remote_log or "")
            statuses = client.query_jobs(job_ids) if job_ids else ()
            reported_by_id = {
                str(_read_member(status, "job_id", "")).strip():
                _normalize_slurm_state(_read_member(status, "state", "UNKNOWN"))
                for status in statuses or ()
                if str(_read_member(status, "job_id", "")).strip()
            }
            scheduler_terminal = bool(job_ids) and all(
                _is_terminal_state(reported_by_id.get(job_id, "UNKNOWN"))
                for job_id in job_ids
            )
            completion = None
            completion_error = ""
            if scheduler_terminal:
                if not remote_run_dir:
                    completion_error = (
                        "The remote run directory is unknown, so its output "
                        "inventory could not be verified."
                    )
                elif not run.remote_cli:
                    completion_error = (
                        "The remote hpc_bundle path is unknown, so its output "
                        "inventory could not be verified."
                    )
                else:
                    try:
                        completion = client.query_run_completion(
                            run.remote_cli,
                            remote_run_dir,
                            python_executable=values.python_executable,
                        )
                    except Exception as exc:
                        completion_error = str(exc).strip() or type(exc).__name__
            logs: list[str] = []
            if remote_log:
                try:
                    driver_log = client.tail_log(remote_log, lines=200)
                except Exception as exc:
                    logs.append(f"Driver log unavailable: {exc}")
                else:
                    if driver_log:
                        logs.append(str(driver_log))
            tail_tasks = getattr(client, "tail_run_logs", None)
            if remote_run_dir and callable(tail_tasks):
                try:
                    task_logs = tail_tasks(remote_run_dir, lines=80, files=8)
                except Exception as exc:
                    logs.append(f"SLURM task logs unavailable: {exc}")
                else:
                    if task_logs:
                        logs.append(str(task_logs))
            return {
                "stage_result": stage_result,
                "statuses": statuses,
                "completion": completion,
                "completion_error": completion_error,
                "log": "\n".join(logs),
            }

        return self._start_operation(
            "refresh",
            operation,
            lambda result: self._refresh_succeeded(run.run_id, result),
        )

    @Slot()
    def cancel_selected_run(self) -> bool:
        run = self.selected_run()
        if run is None:
            self._show_error("Select one tracked run to cancel.")
            return False
        if not run.job_ids:
            self._show_error("The selected run has no recorded SLURM job IDs.")
            return False
        answer = QMessageBox.question(
            self,
            "Cancel SLURM job?",
            f"Request cancellation of {', '.join(run.job_ids)} for {run.run_id}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self._set_status("Cancellation was not sent; the remote job is unchanged.")
            return False
        values = ConnectionValues(**run.connection)

        def operation() -> Any:
            return self._remote_client(values).cancel(run.job_ids)

        return self._start_operation(
            "cancel", operation, lambda result: self._cancel_succeeded(run.run_id, result)
        )

    @Slot()
    def download_selected_results(self) -> bool:
        run = self.selected_run()
        if run is None:
            self._show_error("Select one tracked run to download.")
            return False
        if not run.remote_results:
            self._show_error("The selected run has no recorded remote results folder.")
            return False
        if not _is_run_terminal_state(run.state):
            self._show_error(
                "Results can be downloaded after SLURM reports a terminal state. "
                "Refresh the selected run first."
            )
            return False
        if run.results_complete is not True:
            self._show_error(
                "The remote run's manifest-exact output inventory has not passed. "
                "Refresh the selected run; downloads remain disabled when outputs "
                "are missing or remote verification fails."
            )
            return False
        if not run.remote_cli or not run.remote_run_dir:
            self._show_error(
                "The remote CLI or run directory is missing, so GRIM cannot "
                "re-verify this run before download. Refresh the selected run."
            )
            return False
        directory = QFileDialog.getExistingDirectory(
            self, "Download HPC Results", ""
        )
        if not directory:
            return False
        local_target = Path(directory) / PurePosixPath(run.remote_results).name
        if local_target.exists():
            self._show_error(
                "The result target already exists. Choose a different parent folder: "
                f"{local_target}"
            )
            return False
        values = ConnectionValues(**run.connection)

        def operation() -> Any:
            client = self._remote_client(values)
            try:
                completion = client.query_run_completion(
                    run.remote_cli,
                    run.remote_run_dir,
                    python_executable=values.python_executable,
                )
            except Exception as exc:
                detail = str(exc).strip() or type(exc).__name__
                raise _DownloadVerificationError(
                    "Fresh remote output verification failed; no files were "
                    f"downloaded: {detail}"
                ) from exc
            if completion.get("complete") is not True:
                missing = completion.get("missing", ())
                unexpected = completion.get("unexpected", ())
                derived_missing = completion.get("derived_missing", ())
                detail = (
                    f"{completion.get('n_done', 0)}/{completion.get('n_units', 0)} "
                    f"unit outputs, {len(missing)} missing, "
                    f"{len(unexpected)} unexpected, and "
                    f"{len(derived_missing)} derived outputs missing"
                )
                raise _DownloadVerificationError(
                    "The fresh remote manifest inventory is incomplete "
                    f"({detail}); no files were downloaded.",
                    completion,
                )
            transfer = client.download_results(run.remote_results, directory)
            return {"completion": completion, "transfer": transfer}

        return self._start_operation(
            "download",
            operation,
            lambda result: self._download_succeeded(run.run_id, directory, result),
            failure_handler=lambda exc: self._download_failed(run.run_id, exc),
        )

    # ------------------------------------------------------------------
    # Background lifecycle and result handling
    # ------------------------------------------------------------------
    def _start_operation(
        self,
        kind: str,
        operation: Callable[[], Any],
        success_handler: Callable[[Any], None],
        *,
        failure_handler: Callable[[Exception], None] | None = None,
    ) -> bool:
        if self.job_is_running():
            self._show_error(f"{self.busy_operation()} is already running.")
            return False
        thread = QThread(self)
        worker = _OperationWorker(operation)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.succeeded.connect(self._operation_succeeded)
        worker.failed.connect(self._operation_failed)
        worker.succeeded.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.succeeded.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._operation_thread_finished)
        self._thread = thread
        self._worker = worker
        self._active_kind = kind
        self._success_handler = success_handler
        self._failure_handler = failure_handler
        self._last_error = ""
        self._set_busy(True)
        self._set_status(
            {
                "export": "Building and verifying the portable HPC bundle…",
                "verify": "Verifying the portable HPC bundle…",
                "test": "Testing the credential-safe HPC connection…",
                "submit": "Verifying, uploading, staging, and submitting the HPC run…",
                "refresh": "Refreshing SLURM state and the remote log…",
                "cancel": "Sending the SLURM cancellation request…",
                "download": "Downloading the selected HPC results…",
            }[kind]
        )
        thread.start()
        return True

    @Slot(object)
    def _operation_succeeded(self, result: Any) -> None:
        handler = self._success_handler
        if handler is None:
            self._show_error("HPC operation completed without a result handler.")
            return
        try:
            handler(result)
        except Exception as exc:
            self._show_error(str(exc))

    @Slot(object)
    def _operation_failed(self, exc: Exception) -> None:
        handler = self._failure_handler
        if handler is None:
            self._show_error(str(exc).strip() or type(exc).__name__)
            return
        try:
            handler(exc)
        except Exception as handler_exc:
            self._show_error(str(handler_exc).strip() or type(handler_exc).__name__)

    @Slot()
    def _operation_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self._active_kind = ""
        self._success_handler = None
        self._failure_handler = None
        self._set_busy(False)

    def _connection_succeeded(self, result: Any) -> None:
        host = _read_member(result, "hostname", "")
        user = _read_member(result, "username", "")
        server = "@".join(value for value in (str(user), str(host)) if value)
        detail = f" ({server})" if server else ""
        command_result = _read_member(result, "result", None)
        output = _read_member(command_result, "stdout", "")
        self._append_log(str(output or result))
        self._set_status(f"Connection successful{detail}.")
        self.save_settings()

    def _bundle_exported(self, path: str, result: Any) -> None:
        bundle_id = str(_read_member(result, "bundle_id", ""))
        self.bundle_path_edit.setText(path)
        self._append_log(json.dumps(_plain_mapping(result), indent=2, default=str))
        self._set_status(
            f"Portable HPC bundle ready: {path}"
            + (f" ({bundle_id})" if bundle_id else "")
        )
        self.save_settings()

    def _bundle_verified(self, path: Path, result: Any) -> None:
        bundle_id = str(_read_member(result, "bundle_id", ""))
        self._append_log(json.dumps(_plain_mapping(result), indent=2, default=str))
        self._set_status(
            f"Portable HPC bundle verified: {path}"
            + (f" ({bundle_id})" if bundle_id else "")
        )
        self.save_settings()

    def _submission_succeeded(self, snapshot: Mapping[str, Any], value: Any) -> None:
        request = _plain_mapping(value.get("request"))
        result_object = value.get("result")
        result = _plain_mapping(result_object)
        # The table uses the user's stable, readable label.  Backend run IDs
        # and bundle UUIDs remain in the staged payload/log for provenance.
        run_id = str(snapshot["run_name"])
        raw_ids = _read_member(result_object, "job_ids", ()) or result.get("job_ids", ())
        job_ids = tuple(str(item).strip() for item in raw_ids if str(item).strip())
        run_dir = str(result.get("run_dir") or "")
        results = _safe_remote_join(run_dir, "results") if run_dir else ""
        remote_log = str(result.get("log_path") or "")
        ok = bool(result.get("ok", True))
        state = (
            "SUBMITTED"
            if ok and job_ids
            else "STAGED"
            if ok
            else "PARTIAL SUBMISSION"
            if job_ids
            else "FAILED"
        )
        tracked = self._tracked_runs.get(run_id) or TrackedRun(
            run_id=run_id,
            solver=str(request.get("solver") or snapshot["request"]["solver"]),
        )
        tracked.solver = str(request.get("solver") or snapshot["request"]["solver"])
        tracked.bundle_id = str(
            result.get("bundle_id")
            or request.get("bundle_id")
            or snapshot["bundle_id"]
        )
        tracked.job_ids = job_ids
        tracked.state = state
        tracked.progress = (
            "Submission accepted"
            if state == "SUBMITTED"
            else "Bundle staged without job IDs"
            if state == "STAGED"
            else "Some jobs were submitted before the driver failed"
            if state == "PARTIAL SUBMISSION"
            else "Remote staging failed"
        )
        tracked.remote_bundle = str(
            _read_member(value.get("transfer"), "remote_path", "")
            or snapshot["remote_bundle"]
        )
        tracked.remote_cli = str(snapshot["remote_cli"])
        tracked.remote_stage_dir = str(
            result.get("stage_dir") or snapshot["remote_stage_dir"]
        )
        tracked.remote_stage_result = str(snapshot["remote_stage_result"])
        tracked.remote_run_dir = run_dir
        tracked.remote_results = results
        tracked.remote_log = remote_log
        tracked.local_bundle = ""
        tracked.updated_utc = _utc_now_text()
        tracked.connection = asdict(snapshot["connection"])
        self._tracked_runs[run_id] = tracked
        self._render_runs(select_run_id=tracked.run_id)
        self._append_log(json.dumps(result, indent=2, default=str))
        ids = ", ".join(job_ids) if job_ids else "none returned"
        self._set_status(f"HPC run {tracked.run_id} staged; SLURM job ID(s): {ids}.")
        self.save_settings()
        self.run_submitted.emit(tracked)

    def _submission_failed(self, snapshot: Mapping[str, Any], exc: Exception) -> None:
        run_id = str(snapshot["run_name"])
        message = str(exc).strip() or type(exc).__name__
        if not bool(snapshot.get("stage_started")):
            self._tracked_runs.pop(run_id, None)
            self._render_runs()
            self.save_settings()
            self._show_error(
                f"{message} No remote staging command was started, so no SLURM "
                "job was submitted."
            )
            return

        tracked = self._tracked_runs.get(run_id) or TrackedRun(
            run_id=run_id,
            solver=str(snapshot["request"]["solver"]),
        )
        bundle_result = getattr(exc, "bundle_result", None)
        result = _plain_mapping(bundle_result)
        raw_ids = _read_member(bundle_result, "job_ids", ()) or result.get(
            "job_ids", ()
        )
        job_ids = tuple(str(item).strip() for item in raw_ids if str(item).strip())
        run_dir = str(result.get("run_dir") or tracked.remote_run_dir or "")
        tracked.bundle_id = str(
            result.get("bundle_id") or snapshot["bundle_id"]
        )
        tracked.job_ids = job_ids or tracked.job_ids
        tracked.state = (
            "PARTIAL SUBMISSION"
            if tracked.job_ids
            else "FAILED"
            if result
            else "SUBMISSION UNKNOWN"
        )
        tracked.progress = (
            "Some job IDs were recovered; refresh before taking further action"
            if tracked.job_ids
            else "Remote stage result reported a failure"
            if result
            else "Connection ended during staging; refresh to recover the result"
        )
        tracked.remote_bundle = str(snapshot["remote_bundle"])
        tracked.remote_cli = str(snapshot["remote_cli"])
        tracked.remote_stage_dir = str(
            result.get("stage_dir") or snapshot["remote_stage_dir"]
        )
        tracked.remote_stage_result = str(snapshot["remote_stage_result"])
        tracked.remote_run_dir = run_dir
        tracked.remote_results = (
            _safe_remote_join(run_dir, "results") if run_dir else ""
        )
        tracked.remote_log = str(result.get("log_path") or tracked.remote_log or "")
        tracked.updated_utc = _utc_now_text()
        tracked.connection = asdict(snapshot["connection"])
        self._tracked_runs[run_id] = tracked
        if result:
            self._append_log(json.dumps(result, indent=2, default=str))
        self._render_runs(select_run_id=run_id)
        self.save_settings()
        self._show_error(
            f"{message} Run {run_id} is marked {tracked.state}; use Refresh to "
            "recover stage_result.json instead of resubmitting it."
        )
        if tracked.job_ids:
            self.run_submitted.emit(tracked)

    def _refresh_succeeded(self, run_id: str, value: Mapping[str, Any]) -> None:
        run = self._tracked_runs[run_id]
        stage_object = value.get("stage_result")
        if stage_object is not None:
            stage = _plain_mapping(stage_object)
            raw_ids = _read_member(stage_object, "job_ids", ()) or stage.get(
                "job_ids", ()
            )
            recovered_ids = tuple(
                str(item).strip() for item in raw_ids if str(item).strip()
            )
            if recovered_ids:
                run.job_ids = recovered_ids
            run.bundle_id = str(stage.get("bundle_id") or run.bundle_id)
            run.remote_stage_dir = str(
                stage.get("stage_dir") or run.remote_stage_dir
            )
            recovered_run_dir = str(stage.get("run_dir") or run.remote_run_dir or "")
            run.remote_run_dir = recovered_run_dir
            run.remote_results = (
                _safe_remote_join(recovered_run_dir, "results")
                if recovered_run_dir
                else ""
            )
            run.remote_log = str(stage.get("log_path") or run.remote_log or "")
            if not bool(stage.get("ok", False)):
                stage_state = str(stage.get("stage_state") or "").lower()
                run.state = (
                    "PARTIAL SUBMISSION"
                    if run.job_ids
                    else "SUBMISSION UNKNOWN"
                    if stage_state == "running"
                    else "FAILED"
                )
                run.progress = (
                    "Recovered submitted job IDs from an interrupted stage"
                    if run.job_ids
                    else "Remote stage has not finalized; refresh again before resubmitting"
                    if stage_state == "running"
                    else "Recovered a failed remote stage result"
                )
            elif run.job_ids:
                run.state = "SUBMITTED"
                run.progress = "Recovered submitted job IDs from remote stage result"
            else:
                run.state = "STAGED"
                run.progress = "Recovered remote stage result without job IDs"
        statuses = value.get("statuses", ())
        if isinstance(statuses, Mapping):
            statuses = tuple(statuses.values())
        reported_pairs: list[tuple[str, str]] = []
        for status in statuses or ():
            job_id = str(_read_member(status, "job_id", "")).strip()
            state = _normalize_slurm_state(
                _read_member(status, "state", "UNKNOWN")
            )
            reported_pairs.append((job_id, state))

        # Aggregate only across the authoritative IDs recorded for this run.
        # Slurm may return extra step rows (for example ``123.batch``), and an
        # absent row must remain UNKNOWN rather than allowing one completed or
        # failed row to terminalize the whole multi-job run.
        reported_by_id = {
            job_id: state for job_id, state in reported_pairs if job_id
        }
        expected_ids = set(run.job_ids)
        pairs = [
            (job_id, reported_by_id.get(job_id, "UNKNOWN"))
            for job_id in run.job_ids
        ]
        extra_pairs = [
            pair for pair in reported_pairs if pair[0] not in expected_ids
        ]
        states = [state for _job_id, state in pairs]
        if states:
            all_terminal = all(state in _TERMINAL_STATES for state in states)
            if all_terminal:
                if all(state == "COMPLETED" for state in states):
                    run.state = "COMPLETED"
                else:
                    run.state = next(
                        state
                        for state in states
                        if state in _TERMINAL_STATES - {"COMPLETED"}
                    )
            elif any(state == "RUNNING" for state in states):
                run.state = "RUNNING"
            elif any(state == "PENDING" for state in states):
                run.state = "PENDING"
            else:
                active_states = [
                    state
                    for state in states
                    if state != "UNKNOWN" and state not in _TERMINAL_STATES
                ]
                run.state = (
                    active_states[0] if active_states else "STATUS INCOMPLETE"
                )
            run.progress = ", ".join(
                f"{job_id}: {state}" if job_id else state
                for job_id, state in (*pairs, *extra_pairs)
            )
            if all_terminal:
                scheduler_state = run.state
                scheduler_progress = run.progress
                completion = _plain_mapping(value.get("completion"))
                completion_error = str(value.get("completion_error") or "").strip()
                expected = completion.get("n_units", 0)
                present = completion.get("n_done", 0)
                run.results_expected = (
                    expected if type(expected) is int and expected >= 0 else 0
                )
                run.results_present = (
                    present if type(present) is int and present >= 0 else 0
                )
                run.results_complete = completion.get("complete") is True
                if run.results_complete:
                    run.progress = (
                        f"{scheduler_progress}; remote manifest verified "
                        f"({run.results_present}/{run.results_expected} unit outputs)"
                    )
                else:
                    run.state = "INCOMPLETE"
                    if completion_error:
                        result_detail = (
                            "remote output verification failed: " + completion_error
                        )
                    elif completion:
                        missing = completion.get("missing", ())
                        unexpected = completion.get("unexpected", ())
                        details = [
                            f"{run.results_present}/{run.results_expected} unit outputs"
                        ]
                        if isinstance(missing, (list, tuple)) and missing:
                            details.append(f"{len(missing)} missing")
                        if isinstance(unexpected, (list, tuple)) and unexpected:
                            details.append(f"{len(unexpected)} unexpected")
                        derived_missing = completion.get("derived_missing", ())
                        derived_unexpected = completion.get(
                            "derived_unexpected", ()
                        )
                        if (
                            isinstance(derived_missing, (list, tuple))
                            and derived_missing
                        ):
                            details.append(
                                f"{len(derived_missing)} derived outputs missing"
                            )
                        if (
                            isinstance(derived_unexpected, (list, tuple))
                            and derived_unexpected
                        ):
                            details.append(
                                f"{len(derived_unexpected)} derived outputs unexpected"
                            )
                        attestation_error = str(
                            completion.get("attestation_error") or ""
                        ).strip()
                        publication_error = str(
                            completion.get("publication_error") or ""
                        ).strip()
                        if attestation_error:
                            details.append("unit attestation failed: " + attestation_error)
                        if publication_error:
                            details.append("publication check failed: " + publication_error)
                        result_detail = (
                            "remote manifest incomplete ("
                            + ", ".join(details)
                            + ")"
                        )
                    else:
                        result_detail = "remote output verification returned no evidence"
                    run.progress = (
                        f"SLURM {scheduler_state}: {scheduler_progress}; {result_detail}"
                    )
            else:
                # A prior successful inventory must never remain authoritative
                # if Slurm later reports the run active or incompletely known.
                run.results_complete = None
                run.results_expected = 0
                run.results_present = 0
        elif stage_object is None:
            run.progress = "No scheduler record returned"
            run.results_complete = None
            run.results_expected = 0
            run.results_present = 0
        run.updated_utc = _utc_now_text()
        log = value.get("log", "")
        if log:
            self._append_log(str(log))
        self._render_runs(select_run_id=run_id)
        self._set_status(f"{run_id}: {run.state}. {run.progress}")
        self.save_settings()

    def _cancel_succeeded(self, run_id: str, result: Any) -> None:
        run = self._tracked_runs[run_id]
        run.state = "CANCEL REQUESTED"
        run.progress = "Awaiting scheduler refresh"
        run.updated_utc = _utc_now_text()
        self._append_log(str(result))
        self._render_runs(select_run_id=run_id)
        self._set_status(f"Cancellation requested for {run_id}; refresh to confirm.")
        self.save_settings()

    def _download_succeeded(self, run_id: str, directory: str, result: Any) -> None:
        payload = _plain_mapping(result)
        completion = _plain_mapping(payload.get("completion"))
        transfer = payload.get("transfer", result)
        run = self._tracked_runs[run_id]
        run.results_complete = completion.get("complete") is True
        expected = completion.get("n_units", 0)
        present = completion.get("n_done", 0)
        run.results_expected = expected if type(expected) is int and expected >= 0 else 0
        run.results_present = present if type(present) is int and present >= 0 else 0
        run.updated_utc = _utc_now_text()
        self._append_log(str(transfer))
        local_path = str(
            _read_member(transfer, "local_path", "")
            or (Path(directory) / "results")
        )
        self._render_runs(select_run_id=run_id)
        self.save_settings()
        self._set_status(f"Downloaded {run_id} results to {local_path}.")
        self.results_downloaded.emit(local_path)

    def _download_failed(self, run_id: str, exc: Exception) -> None:
        message = str(exc).strip() or type(exc).__name__
        if isinstance(exc, _DownloadVerificationError):
            run = self._tracked_runs[run_id]
            prior_state = run.state
            completion = exc.completion
            run.state = "INCOMPLETE"
            run.results_complete = False
            expected = completion.get("n_units", 0)
            present = completion.get("n_done", 0)
            run.results_expected = (
                expected if type(expected) is int and expected >= 0 else 0
            )
            run.results_present = (
                present if type(present) is int and present >= 0 else 0
            )
            run.progress = f"SLURM {prior_state}; download blocked: {message}"
            run.updated_utc = _utc_now_text()
            self._render_runs(select_run_id=run_id)
            self.save_settings()
        self._show_error(message)

    # ------------------------------------------------------------------
    # Small UI helpers
    # ------------------------------------------------------------------
    def _render_runs(self, *, select_run_id: str | None = None) -> None:
        selected = select_run_id
        if selected is None:
            current = self.selected_run()
            selected = current.run_id if current is not None else None
        runs = list(self._tracked_runs.values())
        self.jobs_table.setRowCount(len(runs))
        for row, run in enumerate(runs):
            values = (
                run.run_id,
                run.solver.upper(),
                ", ".join(run.job_ids),
                run.state,
                run.progress,
                run.remote_run_dir or run.remote_bundle,
                run.updated_utc,
            )
            for column, text in enumerate(values):
                item = QTableWidgetItem(str(text))
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, run.run_id)
                self.jobs_table.setItem(row, column, item)
            if run.run_id == selected:
                self.jobs_table.selectRow(row)
        self._run_selection_changed()

    def _run_selection_changed(self) -> None:
        selected = self.selected_run()
        has_run = selected is not None and not self.job_is_running()
        self.btn_refresh.setEnabled(has_run)
        self.btn_cancel.setEnabled(
            bool(
                has_run
                and selected
                and selected.job_ids
                and not _is_run_terminal_state(selected.state)
            )
        )
        self.btn_download.setEnabled(
            bool(
                has_run
                and selected
                and selected.remote_results
                and _is_run_terminal_state(selected.state)
                and selected.results_complete is True
            )
        )

    def _set_busy(self, busy: bool) -> None:
        self.controls_content.setEnabled(not busy)
        self.btn_refresh.setEnabled(False if busy else self.selected_run() is not None)
        self.btn_cancel.setEnabled(False)
        self.btn_download.setEnabled(False)
        if not busy:
            self._run_selection_changed()

    def _set_status(self, message: str) -> None:
        text = str(message).strip() or "Ready."
        self.status_label.setText(text)
        self.status_changed.emit(text)

    def _show_error(self, message: str) -> None:
        text = str(message).strip() or "HPC operation failed."
        self._last_error = text
        self._append_log(f"ERROR: {text}")
        self._set_status(text)

    def _append_log(self, text: str) -> None:
        value = str(text).strip()
        if not value:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{stamp}] {value}")

    def _transport_changed(self) -> None:
        putty = self.transport_combo.currentData() == "putty"
        self.profile_edit.setPlaceholderText(
            "Required saved PuTTY session name" if putty else "Optional SSH config alias"
        )
        self.putty_host_key_edit.setEnabled(putty)
        self.connection_hint.setText(
            (
                "PuTTY mode uses Plink in batch mode and a saved session. "
                "Authenticate through Pageant or the session's approved key."
            )
            if putty
            else (
                "OpenSSH uses an SSH config alias, Windows ssh-agent, or the "
                "selected identity file. Unknown host keys are rejected."
            )
        )
        self._profile_changed()

    def _profile_changed(self) -> None:
        putty = self.transport_combo.currentData() == "putty"
        alias = bool(self.profile_edit.text().strip()) and not putty
        direct = not putty and not alias
        self.host_edit.setEnabled(direct)
        self.username_edit.setEnabled(direct)
        self.port_spin.setEnabled(direct)
        self.identity_edit.setEnabled(direct)
        self.btn_identity.setEnabled(direct)
        if alias:
            self.connection_hint.setText(
                "OpenSSH config-alias mode is active. Host, user, port, and key "
                "come from that profile and the direct fields below are ignored."
            )

    def _solver_changed(self) -> None:
        bor = self.solver_combo.currentData() == "bor"
        self.elevation_label.setVisible(bor)
        self.elevation_edit.setVisible(bor)
        self.body_axis_az_label.setVisible(bor)
        self.body_axis_az_spin.setVisible(bor)
        self.body_axis_el_label.setVisible(bor)
        self.body_axis_el_spin.setVisible(bor)
        self.body_roll_label.setVisible(bor)
        self.body_roll_spin.setVisible(bor)
        self.btn_add_frd.setVisible(not bor)
        self.btn_add_opn.setVisible(not bor)
        self.btn_add_bor.setVisible(bor)

    def _choose_identity(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose SSH identity file", self.identity_edit.text().strip()
        )
        if path:
            self.identity_edit.setText(path)

    def _choose_geometries(self, role: str) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, f"Add {role} geometries", "", "GHOST geometry (*.geo)"
        )
        added = 0
        try:
            for path in paths:
                added += int(self.add_geometry(role, path))
        except Exception as exc:
            self._show_error(str(exc))
            return
        if paths:
            self._set_status(f"Added {added} {role} geometry file(s).")

    def _remove_selected_geometries(self) -> None:
        rows = sorted(
            {index.row() for index in self.geometry_list.selectedIndexes()}, reverse=True
        )
        for row in rows:
            self.geometry_list.takeItem(row)
        if rows:
            self._set_status(f"Removed {len(rows)} geometry file(s).")

    def _choose_bundle(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Choose Existing Bundle or Parent Folder for New Bundle",
            self.bundle_path_edit.text().strip(),
        )
        if not path:
            return
        selected = Path(path)
        if (selected / "request.json").is_file():
            target = selected
        else:
            label = self.run_name_edit.text().strip()
            if not _SAFE_RUN_NAME.fullmatch(label):
                label = "grim_hpc"
            base = selected / f"{label}_bundle"
            target = base
            suffix = 2
            while target.exists():
                target = selected / f"{base.name}_{suffix}"
                suffix += 1
        self.bundle_path_edit.setText(str(target))


__all__ = [
    "ConnectionValues",
    "RunsWorkspace",
    "TrackedRun",
    "_parse_number_list",
]
