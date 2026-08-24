"""Compact, non-blocking controls for coherent feature assembly.

The physics and placement validation deliberately do not live here.
``FeatureWorkflowAdapter`` accepts the authoritative GHOST
``feature_workflow`` module (or a compatible injected service), while
``FeatureAssemblyFormModel`` keeps request construction testable without Qt or
GHOST on the import path.  The fixed CSV headers are mirrored here only so the
GUI can explain the contract and write blank templates before a backend is
connected; GHOST remains the authoritative parser.

Preview visibility is intentionally absent from the request model.  Hiding a
point or line group in the Assembly 3-D view must never remove that response
from the coherent physical assembly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable


UNIT_CHOICES = (
    ("inches (in)", "inches"),
    ("millimetres (mm)", "millimeters"),
    ("metres (m)", "meters"),
    ("feet (ft)", "feet"),
)


# Display/template mirrors of GHOST's versioned point- and line-placement v1
# contracts.  These are intentionally strict: the GUI, local scripts, and HPC
# workflow all accept the same files without column inference or conversion.
POINT_PLACEMENT_COLUMNS = (
    "placement_id",
    "dataset_id",
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "roll_x",
    "roll_y",
    "roll_z",
)
LINE_PLACEMENT_COLUMNS = (
    "line_id",
    "dataset_id",
    "segment_index",
    "x1",
    "y1",
    "z1",
    "x2",
    "y2",
    "z2",
    "n1x",
    "n1y",
    "n1z",
    "n2x",
    "n2y",
    "n2z",
)
POINT_PLACEMENT_EXAMPLE = (
    "fastener_001,fastener,1.2,8.4,0.5,0,0,1,1,0,0"
)
LINE_PLACEMENT_EXAMPLE = (
    "gap_001,panel_gap,1,-2,6,0,-2,10,0,0,0,1,0,0,1"
)


def placement_csv_template_text(kind: str) -> str:
    """Return the exact blank v1 placement template for ``kind``."""

    normalized = str(kind).strip().lower()
    if normalized == "point":
        columns = POINT_PLACEMENT_COLUMNS
    elif normalized == "line":
        columns = LINE_PLACEMENT_COLUMNS
    else:
        raise ValueError("Placement template kind must be 'point' or 'line'.")
    return ",".join(columns) + "\n"


def write_placement_csv_template(kind: str, path: str | Path) -> Path:
    """Write a blank strict placement CSV and return its final path."""

    raw = _clean_path(path)
    if not raw:
        raise ValueError("Choose where to save the placement CSV template.")
    target = Path(raw)
    if not target.suffix:
        target = target.with_suffix(".csv")
    target.write_text(placement_csv_template_text(kind), encoding="utf-8")
    return target


@runtime_checkable
class _FeatureWorkflowModule(Protocol):
    FeatureAssemblyRequest: Callable[..., Any]

    def discover_feature_dataset_ids(self, **kwargs: Any) -> Any: ...

    def prepare_feature_assembly(self, request: Any) -> Any: ...

    def execute_feature_assembly(self, plan: Any) -> Any: ...


@dataclass(frozen=True)
class FeatureWorkflowAdapter:
    """Neutral adapter around the authoritative feature-workflow API.

    Pass ``FeatureWorkflowAdapter.from_module(feature_workflow)`` to the panel.
    Keeping these four callables explicit avoids importing GHOST from GRIM and
    makes dependency injection straightforward in tests and packaged builds.
    """

    request_factory: Callable[..., Any]
    discover: Callable[..., Any]
    prepare: Callable[[Any], Any]
    execute: Callable[[Any], Any]
    preview_inputs: Callable[..., Any] | None = None

    @classmethod
    def from_module(cls, module: _FeatureWorkflowModule) -> "FeatureWorkflowAdapter":
        missing = [
            name
            for name in (
                "FeatureAssemblyRequest",
                "discover_feature_dataset_ids",
                "prepare_feature_assembly",
                "execute_feature_assembly",
            )
            if not callable(getattr(module, name, None))
        ]
        if missing:
            raise TypeError(
                "Feature workflow service is missing callable(s): "
                + ", ".join(missing)
            )
        mirrored_contracts = (
            ("POINT_CSV_COLUMNS", POINT_PLACEMENT_COLUMNS),
            ("LINE_CSV_COLUMNS", LINE_PLACEMENT_COLUMNS),
        )
        for name, expected in mirrored_contracts:
            backend_columns = getattr(module, name, None)
            if backend_columns is not None and tuple(backend_columns) != expected:
                raise RuntimeError(
                    f"GHOST {name} no longer matches the placement format "
                    "shown by GRIM. Update the GUI template before assembly."
                )
        return cls(
            request_factory=module.FeatureAssemblyRequest,
            discover=module.discover_feature_dataset_ids,
            prepare=module.prepare_feature_assembly,
            execute=module.execute_feature_assembly,
            preview_inputs=getattr(module, "prepare_feature_input_preview", None),
        )

    @classmethod
    def from_service(cls, service: Any) -> "FeatureWorkflowAdapter":
        """Adapt GRIM's small integration service contract.

        The integration-facing names are ``make_request``,
        ``discover_dataset_ids(point_csv=None, line_csv=None)``, ``prepare``,
        and ``execute``.  ``from_module`` remains available when the GHOST
        module itself is injected directly.
        """

        missing = [
            name
            for name in ("make_request", "discover_dataset_ids", "prepare", "execute")
            if not callable(getattr(service, name, None))
        ]
        if missing:
            raise TypeError(
                "Feature assembly service is missing callable(s): "
                + ", ".join(missing)
            )

        def discover(**kwargs: Any) -> Any:
            return service.discover_dataset_ids(
                point_csv=kwargs.get("point_locations_csv"),
                line_csv=kwargs.get("line_locations_csv"),
            )

        return cls(
            request_factory=service.make_request,
            discover=discover,
            prepare=service.prepare,
            execute=service.execute,
            preview_inputs=getattr(service, "prepare_input_preview", None),
        )


def coerce_feature_workflow(service: Any) -> FeatureWorkflowAdapter:
    """Return an adapter for an adapter or feature-workflow-like module."""

    if isinstance(service, FeatureWorkflowAdapter):
        return service
    if service is None:
        raise RuntimeError(
            "Feature assembly is unavailable because no GHOST feature service "
            "has been connected."
        )
    if callable(getattr(service, "make_request", None)):
        return FeatureWorkflowAdapter.from_service(service)
    return FeatureWorkflowAdapter.from_module(service)


@dataclass
class FeatureAssemblyValues:
    """User-editable values, independent of any GUI toolkit."""

    base_grim: str = ""
    output_grim: str = ""
    coordinate_units: str = "inches"
    surface_mesh: str = ""
    surface_units: str = "inches"
    flip_surface_normals: bool = False
    shadow: bool = False
    shadow_bias_m: float | None = None
    point_locations_csv: str = ""
    line_locations_csv: str = ""
    skin_tol_m: float = 1.0e-3
    skin_phase_tol_deg: float = 15.0
    normal_tol_deg: float = 15.0
    base_dir: str | None = None
    point_datasets: dict[str, str] = field(default_factory=dict)
    line_datasets: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureBuildDispatch:
    """Result returned by the combined prepare/execute operation."""

    plan: Any
    output_path: str


def _clean_path(value: Any) -> str:
    return str(value or "").strip()


def _require_finite_nonnegative(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return number


def _requirements_ids(requirements: Any, attribute: str) -> tuple[str, ...]:
    if isinstance(requirements, Mapping):
        values = requirements.get(attribute, ())
    else:
        values = getattr(requirements, attribute, ())
    ordered = tuple(dict.fromkeys(str(value).strip() for value in values))
    if any(not value for value in ordered):
        raise ValueError("Placement CSV returned an empty dataset_id.")
    return ordered


class FeatureAssemblyFormModel:
    """Headless state, validation, discovery, and service dispatch."""

    def __init__(self, values: FeatureAssemblyValues | None = None) -> None:
        self.values = values if values is not None else FeatureAssemblyValues()
        self._point_dataset_ids: tuple[str, ...] = ()
        self._line_dataset_ids: tuple[str, ...] = ()
        self._point_requirements_csv = ""
        self._line_requirements_csv = ""

    @property
    def point_dataset_ids(self) -> tuple[str, ...]:
        return self._point_dataset_ids

    @property
    def line_dataset_ids(self) -> tuple[str, ...]:
        return self._line_dataset_ids

    def update_dataset_requirements(self, requirements: Any) -> None:
        """Apply discovered IDs while preserving paths for surviving IDs."""

        point_ids = _requirements_ids(requirements, "point_dataset_ids")
        line_ids = _requirements_ids(requirements, "line_dataset_ids")
        self._point_dataset_ids = point_ids
        self._line_dataset_ids = line_ids
        self._point_requirements_csv = _clean_path(
            self.values.point_locations_csv
        )
        self._line_requirements_csv = _clean_path(
            self.values.line_locations_csv
        )
        self.values.point_datasets = {
            dataset_id: _clean_path(self.values.point_datasets.get(dataset_id))
            for dataset_id in point_ids
        }
        self.values.line_datasets = {
            dataset_id: _clean_path(self.values.line_datasets.get(dataset_id))
            for dataset_id in line_ids
        }

    def invalidate_dataset_requirements(self, kind: str | None = None) -> None:
        """Discard IDs that no longer describe the selected/on-disk CSV."""

        normalized = None if kind is None else str(kind).strip().lower()
        if normalized not in (None, "point", "line"):
            raise ValueError("Dataset requirement kind must be point, line, or None.")
        if normalized in (None, "point"):
            self._point_dataset_ids = ()
            self._point_requirements_csv = ""
            self.values.point_datasets = {}
        if normalized in (None, "line"):
            self._line_dataset_ids = ()
            self._line_requirements_csv = ""
            self.values.line_datasets = {}

    def query_dataset_ids(self, service: Any) -> Any:
        """Validate selected CSVs and return IDs without mutating this model."""

        adapter = coerce_feature_workflow(service)
        point_csv = _clean_path(self.values.point_locations_csv)
        line_csv = _clean_path(self.values.line_locations_csv)
        if not point_csv and not line_csv:
            raise ValueError("Select a point or line placement CSV first.")
        return adapter.discover(
            point_locations_csv=point_csv or None,
            line_locations_csv=line_csv or None,
            base_dir=self.values.base_dir,
        )

    def discover_dataset_ids(self, service: Any) -> Any:
        """Ask the authoritative parser to validate CSVs and apply their IDs."""

        requirements = self.query_dataset_ids(service)
        self.update_dataset_requirements(requirements)
        return requirements

    def set_point_dataset(self, dataset_id: str, path: str) -> None:
        self._set_dataset("point", dataset_id, path)

    def set_line_dataset(self, dataset_id: str, path: str) -> None:
        self._set_dataset("line", dataset_id, path)

    def _set_dataset(self, kind: str, dataset_id: str, path: str) -> None:
        key = str(dataset_id).strip()
        ids = self._point_dataset_ids if kind == "point" else self._line_dataset_ids
        if key not in ids:
            raise KeyError(f"Unknown {kind} dataset_id {key!r}.")
        mapping = (
            self.values.point_datasets
            if kind == "point"
            else self.values.line_datasets
        )
        mapping[key] = _clean_path(path)

    def missing_dataset_mappings(self) -> tuple[str, ...]:
        missing = [
            f"point:{dataset_id}"
            for dataset_id in self._point_dataset_ids
            if not _clean_path(self.values.point_datasets.get(dataset_id))
        ]
        missing.extend(
            f"line:{dataset_id}"
            for dataset_id in self._line_dataset_ids
            if not _clean_path(self.values.line_datasets.get(dataset_id))
        )
        return tuple(missing)

    def validate(self) -> None:
        values = self.values
        if not _clean_path(values.base_grim):
            raise ValueError("Select the clean-body/base GRIM file.")
        if not _clean_path(values.output_grim):
            raise ValueError("Choose an output GRIM file.")

        point_csv = _clean_path(values.point_locations_csv)
        line_csv = _clean_path(values.line_locations_csv)
        if not point_csv and not line_csv:
            raise ValueError("Select a point or line placement CSV.")
        if point_csv and point_csv != self._point_requirements_csv:
            raise ValueError(
                "The point CSV changed after its last successful scan. "
                "Re-scan it before continuing."
            )
        if line_csv and line_csv != self._line_requirements_csv:
            raise ValueError(
                "The line CSV changed after its last successful scan. "
                "Re-scan it before continuing."
            )
        if point_csv and not self._point_dataset_ids:
            raise ValueError(
                "Point dataset IDs have not been discovered. Re-scan the "
                "point CSV before continuing."
            )
        if line_csv and not self._line_dataset_ids:
            raise ValueError(
                "Line dataset IDs have not been discovered. Re-scan the line "
                "CSV before continuing."
            )
        missing = self.missing_dataset_mappings()
        if missing:
            raise ValueError(
                "Choose an OPN-FRD GRIM response for: " + ", ".join(missing)
            )
        if values.shadow and not _clean_path(values.surface_mesh):
            raise ValueError(
                "Geometric shadowing requires an STL or facet surface mesh."
            )
        if values.coordinate_units not in {value for _, value in UNIT_CHOICES}:
            raise ValueError(f"Unsupported coordinate units: {values.coordinate_units!r}.")
        if values.surface_units not in {value for _, value in UNIT_CHOICES}:
            raise ValueError(f"Unsupported surface units: {values.surface_units!r}.")
        _require_finite_nonnegative(values.skin_tol_m, "Skin distance tolerance")
        _require_finite_nonnegative(
            values.skin_phase_tol_deg, "Skin phase tolerance"
        )
        normal = _require_finite_nonnegative(
            values.normal_tol_deg, "Normal tolerance"
        )
        if normal > 180.0:
            raise ValueError("Normal tolerance must not exceed 180 degrees.")
        if values.shadow_bias_m is not None:
            _require_finite_nonnegative(values.shadow_bias_m, "Shadow bias")

    def build_request(self, service: Any) -> Any:
        """Create the backend request only after local completeness checks."""

        adapter = coerce_feature_workflow(service)
        self.validate()
        values = self.values
        return adapter.request_factory(
            base_grim=_clean_path(values.base_grim),
            output_grim=_clean_path(values.output_grim),
            coordinate_units=values.coordinate_units,
            surface_mesh=_clean_path(values.surface_mesh) or None,
            surface_units=values.surface_units,
            flip_surface_normals=bool(values.flip_surface_normals),
            shadow=bool(values.shadow),
            shadow_bias_m=(
                None
                if values.shadow_bias_m is None
                else float(values.shadow_bias_m)
            ),
            point_locations_csv=_clean_path(values.point_locations_csv) or None,
            point_datasets={
                key: _clean_path(values.point_datasets[key])
                for key in self._point_dataset_ids
            },
            line_locations_csv=_clean_path(values.line_locations_csv) or None,
            line_datasets={
                key: _clean_path(values.line_datasets[key])
                for key in self._line_dataset_ids
            },
            skin_tol_m=float(values.skin_tol_m),
            skin_phase_tol_deg=float(values.skin_phase_tol_deg),
            normal_tol_deg=float(values.normal_tol_deg),
            base_dir=values.base_dir,
        )

    def prepare_preview(self, service: Any) -> Any:
        adapter = coerce_feature_workflow(service)
        return adapter.prepare(self.build_request(adapter))

    def prepare_input_preview(self, service: Any) -> Any:
        """Preview selected geometry/locations before response mapping.

        This is a deliberately non-physical staging preview.  The optional
        backend callable owns CSV parsing and geometry loading; the GUI only
        passes paths and units through unchanged.
        """

        adapter = coerce_feature_workflow(service)
        if not callable(adapter.preview_inputs):
            raise RuntimeError(
                "This GHOST backend does not support staged input preview. "
                "Use Validate Placements & Preview after mapping responses."
            )
        values = self.values
        base_grim = _clean_path(values.base_grim)
        surface_mesh = _clean_path(values.surface_mesh)
        point_csv = _clean_path(values.point_locations_csv)
        line_csv = _clean_path(values.line_locations_csv)
        if not any((base_grim, surface_mesh, point_csv, line_csv)):
            raise ValueError(
                "Choose a clean-body GRIM, body mesh, or placement CSV to preview."
            )
        if values.coordinate_units not in {value for _, value in UNIT_CHOICES}:
            raise ValueError(f"Unsupported coordinate units: {values.coordinate_units!r}.")
        if values.surface_units not in {value for _, value in UNIT_CHOICES}:
            raise ValueError(f"Unsupported surface units: {values.surface_units!r}.")
        return adapter.preview_inputs(
            base_grim=base_grim or None,
            surface_mesh=surface_mesh or None,
            coordinate_units=values.coordinate_units,
            surface_units=values.surface_units,
            point_locations_csv=point_csv or None,
            line_locations_csv=line_csv or None,
            base_dir=values.base_dir,
        )

    def assemble(self, service: Any) -> FeatureBuildDispatch:
        adapter = coerce_feature_workflow(service)
        plan = adapter.prepare(self.build_request(adapter))
        output = adapter.execute(plan)
        return FeatureBuildDispatch(plan=plan, output_path=str(output))


_GUI_IMPORT_ERROR: Exception | None = None
try:  # Keep the model importable on headless/minimal installations.
    from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
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
        QHeaderView,
        QLabel,
        QLineEdit,
        QPushButton,
        QScrollArea,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except (ImportError, RuntimeError) as exc:  # pragma: no cover - environment-specific
    _GUI_IMPORT_ERROR = exc


GUI_AVAILABLE = _GUI_IMPORT_ERROR is None


if GUI_AVAILABLE:

    class _OperationWorker(QObject):
        succeeded = Signal(object)
        failed = Signal(str)

        def __init__(self, operation: Callable[[], Any]) -> None:
            super().__init__()
            self._operation = operation

        @Slot()
        def run(self) -> None:
            try:
                result = self._operation()
            except Exception as exc:  # The UI reports authoritative validation errors.
                self.failed.emit(str(exc) or type(exc).__name__)
            else:
                self.succeeded.emit(result)


    class _PathPicker(QWidget):
        editing_finished = Signal()

        def __init__(
            self,
            *,
            caption: str,
            file_filter: str,
            save: bool = False,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self.caption = caption
            self.file_filter = file_filter
            self.save = bool(save)
            self.edit = QLineEdit(self)
            self.button = QPushButton("Browse…", self)
            self.button.setAutoDefault(False)
            layout = QHBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.edit, 1)
            layout.addWidget(self.button)
            self.button.clicked.connect(self._browse)
            self.edit.editingFinished.connect(self.editing_finished.emit)

        def path(self) -> str:
            return self.edit.text().strip()

        def set_path(self, path: str) -> None:
            self.edit.setText(_clean_path(path))

        def _browse(self) -> None:
            start = self.path()
            if self.save:
                path, _ = QFileDialog.getSaveFileName(
                    self, self.caption, start, self.file_filter
                )
            else:
                path, _ = QFileDialog.getOpenFileName(
                    self, self.caption, start, self.file_filter
                )
            if not path:
                return
            if self.save and Path(path).suffix == "":
                path += ".grim"
            self.set_path(path)
            self.editing_finished.emit()


    class _DatasetMappingEditor(QWidget):
        mapping_changed = Signal()

        def __init__(self, empty_text: str, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._empty_text = empty_text
            self._ids: tuple[str, ...] = ()
            self.table = QTableWidget(0, 3, self)
            self.table.setHorizontalHeaderLabels(
                ["CSV dataset_id", "Required OPN-FRD response (.grim)", ""]
            )
            self.table.setToolTip(
                "Every dataset_id used by the placement CSV must map to the "
                "matching coherent OPN-FRD .grim response."
            )
            self.table.verticalHeader().setVisible(False)
            self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.table.setAlternatingRowColors(True)
            header = self.table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            self.empty_label = QLabel(empty_text, self)
            self.empty_label.setWordWrap(True)
            self.completeness_label = QLabel(self)
            self.completeness_label.setWordWrap(True)
            self.completeness_label.setObjectName("featureMappingStatus")
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.empty_label)
            layout.addWidget(self.table)
            layout.addWidget(self.completeness_label)
            self.table.cellChanged.connect(self._table_changed)
            self.set_dataset_ids(())

        @property
        def dataset_ids(self) -> tuple[str, ...]:
            return self._ids

        def mapping(self) -> dict[str, str]:
            result: dict[str, str] = {}
            for row, dataset_id in enumerate(self._ids):
                item = self.table.item(row, 1)
                result[dataset_id] = "" if item is None else item.text().strip()
            return result

        def missing_ids(self) -> tuple[str, ...]:
            current = self.mapping()
            return tuple(
                dataset_id
                for dataset_id in self._ids
                if not _clean_path(current.get(dataset_id))
            )

        def _update_completeness(self) -> None:
            missing = self.missing_ids()
            if not self._ids:
                self.completeness_label.setText("")
                self.completeness_label.setVisible(False)
            elif missing:
                self.completeness_label.setText(
                    f"○ {len(missing)} response file(s) still required: "
                    + ", ".join(missing)
                )
                self.completeness_label.setVisible(True)
            else:
                self.completeness_label.setText(
                    f"✓ All {len(self._ids)} dataset response(s) mapped."
                )
                self.completeness_label.setVisible(True)

        def _table_changed(self, *_args: Any) -> None:
            self._update_completeness()
            self.mapping_changed.emit()

        def set_dataset_ids(
            self,
            dataset_ids: tuple[str, ...] | list[str],
            mapping: Mapping[str, str] | None = None,
        ) -> None:
            existing = self.mapping() if self._ids else {}
            if mapping is not None:
                existing.update({str(key): _clean_path(value) for key, value in mapping.items()})
            self._ids = tuple(str(value) for value in dataset_ids)
            self.table.blockSignals(True)
            self.table.setRowCount(len(self._ids))
            for row, dataset_id in enumerate(self._ids):
                id_item = QTableWidgetItem(dataset_id)
                id_item.setFlags(
                    id_item.flags() & ~Qt.ItemFlag.ItemIsEditable
                )
                self.table.setItem(row, 0, id_item)
                self.table.setItem(
                    row, 1, QTableWidgetItem(existing.get(dataset_id, ""))
                )
                button = QPushButton("Browse…", self.table)
                button.clicked.connect(
                    lambda _checked=False, key=dataset_id: self._browse(key)
                )
                self.table.setCellWidget(row, 2, button)
            self.table.blockSignals(False)
            has_rows = bool(self._ids)
            self.empty_label.setVisible(not has_rows)
            self.table.setVisible(has_rows)
            self.table.setMinimumHeight(112 if has_rows else 0)
            self._update_completeness()

        def set_path(self, dataset_id: str, path: str) -> None:
            try:
                row = self._ids.index(str(dataset_id))
            except ValueError as exc:
                raise KeyError(f"Unknown dataset_id {dataset_id!r}.") from exc
            self.table.item(row, 1).setText(_clean_path(path))

        def _browse(self, dataset_id: str) -> None:
            current = self.mapping().get(dataset_id, "")
            path, _ = QFileDialog.getOpenFileName(
                self,
                f"Choose OPN-FRD response for {dataset_id}",
                current,
                "GRIM response (*.grim);;All files (*)",
            )
            if path:
                self.set_path(dataset_id, path)


    class FeatureAssemblyPanel(QWidget):
        """New-user-facing feature assembly form with background execution."""

        preview_ready = Signal(object)
        preview_stale = Signal(str)
        feature_built = Signal(str)
        build_failed = Signal(str)
        status_changed = Signal(str)

        def __init__(
            self,
            parent: QWidget | None = None,
            *,
            service: Any = None,
        ) -> None:
            super().__init__(parent)
            self.model = FeatureAssemblyFormModel()
            self._service: Any = service
            self._thread: QThread | None = None
            self._worker: _OperationWorker | None = None
            self._active_kind = ""
            self._discovery_paths: tuple[str, str] | None = None
            self._preview_is_current = False
            self._build_ui()

        def _build_ui(self) -> None:
            outer = QVBoxLayout(self)
            outer.setContentsMargins(6, 6, 6, 6)
            outer.setSpacing(6)

            intro = QLabel(
                "Add compact point features or expanded line features to a "
                "clean-body response. GRIM validates the locations first, "
                "then shows the body and feature locations together before "
                "you assemble the coherent result.",
                self,
            )
            intro.setWordWrap(True)
            outer.addWidget(intro)

            self.workflow_steps_label = QLabel(
                "Workflow:  1 Choose body  →  2 Load placement CSV  →  "
                "3 Match each dataset_id  →  4 Preview in 3-D  →  "
                "5 Assemble and save",
                self,
            )
            self.workflow_steps_label.setWordWrap(True)
            self.workflow_steps_label.setObjectName("featureWorkflowSteps")
            outer.addWidget(self.workflow_steps_label)

            scroll = QScrollArea(self)
            scroll.setObjectName("featureAssemblyScroll")
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setAutoFillBackground(False)
            scroll.viewport().setAutoFillBackground(False)
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            content = QWidget(scroll)
            content.setObjectName("featureAssemblyContent")
            content.setAutoFillBackground(False)
            self.form_content = content
            content_layout = QVBoxLayout(content)
            content_layout.setContentsMargins(0, 0, 0, 0)
            content_layout.setSpacing(7)

            body_group = QGroupBox("1  Body and output", content)
            body_form = QFormLayout(body_group)
            body_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            self.base_picker = _PathPicker(
                caption="Choose clean-body/base GRIM",
                file_filter="GRIM response (*.grim);;All files (*)",
            )
            self.surface_picker = _PathPicker(
                caption="Choose body surface mesh",
                file_filter="Surface mesh (*.stl *.facet);;All files (*)",
            )
            self.output_picker = _PathPicker(
                caption="Save assembled GRIM",
                file_filter="GRIM response (*.grim);;All files (*)",
                save=True,
            )
            self.coordinate_units = QComboBox(body_group)
            self.surface_units = QComboBox(body_group)
            for label, value in UNIT_CHOICES:
                self.coordinate_units.addItem(label, value)
                self.surface_units.addItem(label, value)
            self.flip_normals = QCheckBox("Flip mesh normals", body_group)
            self.shadow = QCheckBox(
                "Apply geometric body shadowing (requires mesh)", body_group
            )
            mesh_options = QWidget(body_group)
            mesh_layout = QVBoxLayout(mesh_options)
            mesh_layout.setContentsMargins(0, 0, 0, 0)
            mesh_layout.addWidget(self.flip_normals)
            mesh_layout.addWidget(self.shadow)
            self.base_picker.setToolTip(
                "Clean-body response to which the point and/or line feature "
                "responses will be coherently added."
            )
            self.surface_picker.setToolTip(
                "Choose the matching STL/facet surface for a 3-D body. Leave "
                "blank when the base GRIM contains an embedded BoR profile."
            )
            self.coordinate_units.setToolTip(
                "Units used by every x/y/z coordinate in both placement CSVs."
            )
            self.surface_units.setToolTip(
                "Units of the selected STL/facet surface, independent of the CSV units."
            )
            body_form.addRow("Clean-body response (.grim):", self.base_picker)
            body_form.addRow("Body surface (.stl/.facet):", self.surface_picker)
            body_form.addRow("Coordinates in CSV are:", self.coordinate_units)
            body_form.addRow("Surface mesh units:", self.surface_units)
            body_form.addRow("Mesh options:", mesh_options)
            body_form.addRow("Save assembled response as:", self.output_picker)
            self.body_preview_help = QLabel(
                "3-D preview source: the selected STL/facet mesh is shown when "
                "provided; otherwise GRIM shows the BoR profile embedded in the "
                "base response. A non-BoR base requires a matching mesh for "
                "placement validation.",
                body_group,
            )
            self.body_preview_help.setWordWrap(True)
            body_form.addRow("", self.body_preview_help)
            content_layout.addWidget(body_group)

            feature_group = QGroupBox("2  Feature locations and responses", content)
            feature_layout = QVBoxLayout(feature_group)
            self.feature_tabs = QTabWidget(feature_group)

            point_page = QWidget(self.feature_tabs)
            point_layout = QVBoxLayout(point_page)
            self.point_csv_picker = _PathPicker(
                caption="Choose point placement CSV",
                file_filter="CSV placement file (*.csv);;All files (*)",
            )
            self.point_csv_picker.setToolTip(
                "Strict GHOST point-placement CSV. This is the same file used "
                "by local scripts and the HPC workflow."
            )
            point_layout.addWidget(QLabel("Point location/orientation CSV:"))
            point_layout.addWidget(self.point_csv_picker)
            self.point_help_label = QLabel(
                "Same strict GHOST CSV used locally and on HPC—there is no "
                "alternate GUI format. The header is followed directly by data "
                "rows; do not add a units row or comments. Coordinate units "
                "come from 'Coordinates in CSV are' in step 1; normal/roll "
                "vectors are unitless. The normal is local +z; the roll vector "
                "defines local +x/azimuth zero. IDs must be unique.",
                point_page,
            )
            self.point_help_label.setWordWrap(True)
            point_layout.addWidget(self.point_help_label)
            point_format_row = QHBoxLayout()
            self.point_format_button = QPushButton(
                "Show exact point CSV format", point_page
            )
            self.point_format_button.setCheckable(True)
            point_format_row.addWidget(self.point_format_button)
            self.point_schema_label = QLabel(
                "Exact header (column order is fixed):\n"
                + ",".join(POINT_PLACEMENT_COLUMNS)
                + "\nExample row:\n"
                + POINT_PLACEMENT_EXAMPLE,
                point_page,
            )
            self.point_schema_label.setWordWrap(True)
            self.point_schema_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self.point_schema_label.setObjectName("featureCsvSchema")
            self.point_template_button = QPushButton(
                "Save blank point CSV template…", point_page
            )
            self.point_template_button.setToolTip(
                "Write the exact required point header to a new .csv file."
            )
            point_format_row.addWidget(self.point_template_button)
            point_format_row.addStretch(1)
            point_layout.addLayout(point_format_row)
            self.point_schema_label.setVisible(False)
            point_layout.addWidget(self.point_schema_label)
            point_response_help = QLabel(
                "After CSV validation, one row appears below for every "
                "dataset_id. Map each row to its matching coherent OPN-FRD "
                ".grim response (VV, HH, and reciprocal cross-polar response).",
                point_page,
            )
            point_response_help.setWordWrap(True)
            point_layout.addWidget(point_response_help)
            self.point_mapping = _DatasetMappingEditor(
                "Choose a point CSV above. GRIM will validate the exact header "
                "and create one required response row per dataset_id.",
                point_page,
            )
            point_layout.addWidget(self.point_mapping)
            self.feature_tabs.addTab(point_page, "Point features")

            line_page = QWidget(self.feature_tabs)
            line_layout = QVBoxLayout(line_page)
            self.line_csv_picker = _PathPicker(
                caption="Choose line placement CSV",
                file_filter="CSV placement file (*.csv);;All files (*)",
            )
            self.line_csv_picker.setToolTip(
                "Strict GHOST ordered-segment line-placement CSV. This is the "
                "same file used by local scripts and the HPC workflow."
            )
            line_layout.addWidget(QLabel("Line path/orientation CSV:"))
            line_layout.addWidget(self.line_csv_picker)
            self.line_help_label = QLabel(
                "Same strict GHOST CSV used locally and on HPC—there is no "
                "alternate GUI format. The header is followed directly by data "
                "rows; do not add a units row or comments. Coordinate units "
                "come from 'Coordinates in CSV are' in step 1; normals are "
                "unitless. Rows for each line_id must be contiguous; "
                "segment_index starts at 1; adjacent segments meet head-to-tail. "
                "Both endpoint normals point outward.",
                line_page,
            )
            self.line_help_label.setWordWrap(True)
            line_layout.addWidget(self.line_help_label)
            line_format_row = QHBoxLayout()
            self.line_format_button = QPushButton(
                "Show exact line CSV format", line_page
            )
            self.line_format_button.setCheckable(True)
            line_format_row.addWidget(self.line_format_button)
            self.line_schema_label = QLabel(
                "Exact header (column order is fixed):\n"
                + ",".join(LINE_PLACEMENT_COLUMNS)
                + "\nExample row:\n"
                + LINE_PLACEMENT_EXAMPLE,
                line_page,
            )
            self.line_schema_label.setWordWrap(True)
            self.line_schema_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self.line_schema_label.setObjectName("featureCsvSchema")
            self.line_template_button = QPushButton(
                "Save blank line CSV template…", line_page
            )
            self.line_template_button.setToolTip(
                "Write the exact required line header to a new .csv file."
            )
            line_format_row.addWidget(self.line_template_button)
            line_format_row.addStretch(1)
            line_layout.addLayout(line_format_row)
            self.line_schema_label.setVisible(False)
            line_layout.addWidget(self.line_schema_label)
            line_response_help = QLabel(
                "After CSV validation, map each dataset_id to the matching "
                "coherent OPN-FRD line response containing TE and TM.",
                line_page,
            )
            line_response_help.setWordWrap(True)
            line_layout.addWidget(line_response_help)
            self.line_mapping = _DatasetMappingEditor(
                "Choose a line CSV above. GRIM will validate the exact header "
                "and create one required response row per dataset_id.",
                line_page,
            )
            line_layout.addWidget(self.line_mapping)
            self.feature_tabs.addTab(line_page, "Line features")
            feature_layout.addWidget(self.feature_tabs)
            self.scan_button = QPushButton(
                "Validate CSV(s) and refresh response rows", feature_group
            )
            self.scan_button.setToolTip(
                "Parse the selected CSVs with the authoritative GHOST parser "
                "and list every response dataset that must be supplied."
            )
            feature_layout.addWidget(self.scan_button)
            content_layout.addWidget(feature_group)

            advanced = QGroupBox("Advanced placement checks", content)
            advanced.setCheckable(True)
            advanced.setChecked(False)
            advanced_form = QFormLayout(advanced)
            advanced_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            self.skin_tol = QDoubleSpinBox(advanced)
            self.skin_tol.setDecimals(6)
            self.skin_tol.setRange(0.0, 1.0e3)
            self.skin_tol.setValue(1.0e-3)
            self.skin_tol.setSuffix(" m")
            self.phase_tol = QDoubleSpinBox(advanced)
            self.phase_tol.setDecimals(2)
            self.phase_tol.setRange(0.0, 1.0e6)
            self.phase_tol.setValue(15.0)
            self.phase_tol.setSuffix("°")
            self.normal_tol = QDoubleSpinBox(advanced)
            self.normal_tol.setDecimals(2)
            self.normal_tol.setRange(0.0, 180.0)
            self.normal_tol.setValue(15.0)
            self.normal_tol.setSuffix("°")
            self.shadow_bias = QLineEdit(advanced)
            self.shadow_bias.setPlaceholderText("Auto (recommended)")
            advanced_form.addRow("Maximum skin distance:", self.skin_tol)
            advanced_form.addRow("Maximum two-way phase error:", self.phase_tol)
            advanced_form.addRow("Maximum normal mismatch:", self.normal_tol)
            advanced_form.addRow("Shadow ray bias (m):", self.shadow_bias)
            content_layout.addWidget(advanced)
            content_layout.addStretch(1)
            scroll.setWidget(content)
            outer.addWidget(scroll, 1)

            self.preview_help_label = QLabel(
                "Preview Inputs shows the selected STL/facet or embedded BoR "
                "with CSV points/lines before response files are mapped. "
                "Validate Placements & Preview then checks the body skin, "
                "normals, and response mapping completeness. Full response "
                "compatibility is checked during assembly. Preview visibility "
                "checkboxes affect only the display, never the coherent build. "
                "The 3-D view draws point locations and line paths; normal and "
                "roll vectors are checked numerically during Validate but are "
                "not drawn yet.",
                self,
            )
            self.preview_help_label.setWordWrap(True)
            outer.addWidget(self.preview_help_label)

            self.status_label = QLabel(
                "Ready — choose a clean-body response and at least one placement CSV.",
                self,
            )
            self.status_label.setObjectName("featureAssemblyStatus")
            self.status_label.setWordWrap(True)
            self.status_label.setFrameShape(QFrame.Shape.StyledPanel)
            self.status_label.setMargin(6)
            outer.addWidget(self.status_label)

            action_row = QVBoxLayout()
            self.input_preview_button = QPushButton("Preview Inputs in 3-D", self)
            self.input_preview_button.setToolTip(
                "Show available body geometry and CSV locations without requiring "
                "response mappings or an output path. This is visual QA only."
            )
            self.preview_button = QPushButton(
                "Validate Placements && Preview", self
            )
            self.preview_button.setToolTip(
                "Validate body skin, normals, and response mapping completeness, then "
                "show the prepared body and features in the 3-D Assembly view."
            )
            self.build_button = QPushButton("Assemble Coherently && Save", self)
            self.build_button.setToolTip(
                "Run the same validation, coherently add every mapped feature, "
                "and save the selected output .grim file."
            )
            self.build_button.setDefault(True)
            action_row.addWidget(self.input_preview_button)
            action_row.addWidget(self.preview_button)
            action_row.addWidget(self.build_button)
            outer.addLayout(action_row)

            self.status_changed.connect(self.status_label.setText)
            self.base_picker.editing_finished.connect(self._base_path_changed)
            self.surface_picker.editing_finished.connect(self._mark_preview_stale)
            self.point_csv_picker.editing_finished.connect(
                lambda: self._placement_csv_changed("point")
            )
            self.line_csv_picker.editing_finished.connect(
                lambda: self._placement_csv_changed("line")
            )
            self.coordinate_units.currentIndexChanged.connect(
                self._mark_preview_stale
            )
            self.surface_units.currentIndexChanged.connect(self._mark_preview_stale)
            self.flip_normals.toggled.connect(self._mark_preview_stale)
            self.shadow.toggled.connect(self._mark_preview_stale)
            self.skin_tol.valueChanged.connect(self._mark_preview_stale)
            self.phase_tol.valueChanged.connect(self._mark_preview_stale)
            self.normal_tol.valueChanged.connect(self._mark_preview_stale)
            self.shadow_bias.editingFinished.connect(self._mark_preview_stale)
            self.point_mapping.mapping_changed.connect(self._mapping_changed)
            self.line_mapping.mapping_changed.connect(self._mapping_changed)
            self.point_format_button.toggled.connect(
                lambda checked: self._toggle_schema_help("point", checked)
            )
            self.line_format_button.toggled.connect(
                lambda checked: self._toggle_schema_help("line", checked)
            )
            self.point_template_button.clicked.connect(
                lambda _checked=False: self._save_template("point")
            )
            self.line_template_button.clicked.connect(
                lambda _checked=False: self._save_template("line")
            )
            self.scan_button.clicked.connect(self.refresh_dataset_ids)
            self.input_preview_button.clicked.connect(self.preview_inputs)
            self.preview_button.clicked.connect(self.validate_and_preview)
            self.build_button.clicked.connect(self.assemble_and_save)

        def set_service(self, service: Any) -> None:
            coerce_feature_workflow(service)  # Fail early with an actionable API error.
            self._service = service

        def service(self) -> Any:
            return self._service

        def set_base_grim(self, path: str) -> None:
            self.base_picker.set_path(path)
            self._base_path_changed()

        def set_surface_mesh(self, path: str) -> None:
            self.surface_picker.set_path(path)
            self._mark_preview_stale()

        def set_point_csv(self, path: str, *, discover: bool = True) -> None:
            self.point_csv_picker.set_path(path)
            if discover:
                self._placement_csv_changed("point")
            else:
                self.model.invalidate_dataset_requirements("point")
                self.point_mapping.set_dataset_ids(())
                self._mark_preview_stale()

        def set_line_csv(self, path: str, *, discover: bool = True) -> None:
            self.line_csv_picker.set_path(path)
            if discover:
                self._placement_csv_changed("line")
            else:
                self.model.invalidate_dataset_requirements("line")
                self.line_mapping.set_dataset_ids(())
                self._mark_preview_stale()

        def set_output_grim(self, path: str) -> None:
            self.output_picker.set_path(path)

        def job_is_running(self) -> bool:
            return bool(self._thread is not None and self._thread.isRunning())

        def is_busy(self) -> bool:
            """Return whether discovery, validation, or assembly is active."""

            return self.job_is_running()

        def can_close(self) -> bool:
            """Closing is safe only after the non-cancellable physics job ends."""

            return not self.is_busy()

        def closeEvent(self, event: Any) -> None:
            if self.is_busy():
                self.status_changed.emit(
                    "Feature validation/assembly is still running; wait before closing."
                )
                event.ignore()
                return
            super().closeEvent(event)

        def _base_path_changed(self) -> None:
            self._mark_preview_stale()
            base = self.base_picker.path()
            if base and not self.output_picker.path():
                source = Path(base)
                suggestion = source.with_name(source.stem + "_features.grim")
                self.output_picker.set_path(str(suggestion))

        @Slot()
        def _mark_preview_stale(self, *_args: Any) -> None:
            if not self._preview_is_current:
                return
            self._preview_is_current = False
            message = (
                "Inputs changed — the 3-D preview is out of date. Preview "
                "inputs again, or validate placements for an authoritative preview."
            )
            self.status_changed.emit(message)
            self.preview_stale.emit(message)

        def _placement_csv_changed(self, kind: str) -> None:
            self._mark_preview_stale()
            self.model.invalidate_dataset_requirements(kind)
            if kind == "point":
                self.point_mapping.set_dataset_ids(())
            else:
                self.line_mapping.set_dataset_ids(())
            self.refresh_dataset_ids()

        def _mapping_changed(self) -> None:
            self._mark_preview_stale()
            missing = [
                f"point:{dataset_id}" for dataset_id in self.point_mapping.missing_ids()
            ]
            missing.extend(
                f"line:{dataset_id}" for dataset_id in self.line_mapping.missing_ids()
            )
            if missing:
                self.status_changed.emit(
                    "Response mapping incomplete — choose an OPN-FRD .grim for: "
                    + ", ".join(missing)
                )
            elif self.point_mapping.dataset_ids or self.line_mapping.dataset_ids:
                self.status_changed.emit(
                    "All discovered dataset IDs are mapped. Next, validate "
                    "placements and inspect them in the 3-D Assembly view."
                )

        def _toggle_schema_help(self, kind: str, checked: bool) -> None:
            if kind == "point":
                button = self.point_format_button
                label = self.point_schema_label
            else:
                button = self.line_format_button
                label = self.line_schema_label
            label.setVisible(bool(checked))
            button.setText(
                ("Hide" if checked else "Show")
                + f" exact {kind} CSV format"
            )

        def _save_template(self, kind: str) -> None:
            default_name = (
                "point_features_template.csv"
                if kind == "point"
                else "line_features_template.csv"
            )
            path, _ = QFileDialog.getSaveFileName(
                self,
                f"Save blank {kind} placement CSV template",
                default_name,
                "CSV placement file (*.csv);;All files (*)",
            )
            if not path:
                return
            try:
                saved = write_placement_csv_template(kind, path)
            except Exception as exc:
                self._show_error(str(exc))
                return
            self.status_changed.emit(
                f"Saved blank {kind} template: {saved}. Add placement rows, "
                "then choose that CSV above to validate it."
            )

        def _pull_values(self) -> None:
            values = self.model.values
            values.base_grim = self.base_picker.path()
            values.output_grim = self.output_picker.path()
            values.coordinate_units = str(self.coordinate_units.currentData())
            values.surface_mesh = self.surface_picker.path()
            values.surface_units = str(self.surface_units.currentData())
            values.flip_surface_normals = self.flip_normals.isChecked()
            values.shadow = self.shadow.isChecked()
            bias = self.shadow_bias.text().strip()
            try:
                values.shadow_bias_m = None if not bias else float(bias)
            except ValueError as exc:
                raise ValueError("Shadow bias must be a number in metres or blank.") from exc
            values.point_locations_csv = self.point_csv_picker.path()
            values.line_locations_csv = self.line_csv_picker.path()
            values.point_datasets = self.point_mapping.mapping()
            values.line_datasets = self.line_mapping.mapping()
            values.skin_tol_m = self.skin_tol.value()
            values.skin_phase_tol_deg = self.phase_tol.value()
            values.normal_tol_deg = self.normal_tol.value()

        def _show_error(self, text: str) -> None:
            message = str(text).strip() or "Feature assembly failed."
            self.status_changed.emit(message)
            self.build_failed.emit(message)

        @Slot()
        def refresh_dataset_ids(self) -> None:
            if self.job_is_running():
                self.status_changed.emit("A feature operation is already running.")
                return
            try:
                self._pull_values()
                if not (
                    self.model.values.point_locations_csv
                    or self.model.values.line_locations_csv
                ):
                    self.model.update_dataset_requirements(
                        {"point_dataset_ids": (), "line_dataset_ids": ()}
                    )
                    self._apply_requirements_to_tables()
                    self.status_changed.emit("Select a placement CSV to discover IDs.")
                    return
                adapter = coerce_feature_workflow(self._service)
            except Exception as exc:
                self._show_error(str(exc))
                return
            self._discovery_paths = (
                self.model.values.point_locations_csv,
                self.model.values.line_locations_csv,
            )
            self._start_operation(
                "discover",
                lambda: self.model.query_dataset_ids(adapter),
            )

        def _apply_requirements_to_tables(self) -> None:
            self.point_mapping.set_dataset_ids(
                self.model.point_dataset_ids, self.model.values.point_datasets
            )
            self.line_mapping.set_dataset_ids(
                self.model.line_dataset_ids, self.model.values.line_datasets
            )

        @Slot()
        def preview_inputs(self) -> None:
            """Show geometry/locations without requiring response mappings."""

            if self.job_is_running():
                self.status_changed.emit("A feature operation is already running.")
                return
            try:
                self._pull_values()
                adapter = coerce_feature_workflow(self._service)
                if not callable(adapter.preview_inputs):
                    raise RuntimeError(
                        "This GHOST backend does not support staged input preview. "
                        "Use Validate Placements & Preview after mapping responses."
                    )
                if not any(
                    (
                        self.model.values.base_grim,
                        self.model.values.surface_mesh,
                        self.model.values.point_locations_csv,
                        self.model.values.line_locations_csv,
                    )
                ):
                    raise ValueError(
                        "Choose a clean-body GRIM, body mesh, or placement CSV "
                        "to preview."
                    )
            except Exception as exc:
                self._show_error(str(exc))
                return
            self._start_operation(
                "input_preview", lambda: self.model.prepare_input_preview(adapter)
            )

        @Slot()
        def validate_and_preview(self) -> None:
            if self.job_is_running():
                self.status_changed.emit("A feature operation is already running.")
                return
            try:
                self._pull_values()
                adapter = coerce_feature_workflow(self._service)
                # Request construction is quick and catches missing mappings
                # before the authoritative prepare operation is dispatched.
                self.model.build_request(adapter)
            except Exception as exc:
                self._show_error(str(exc))
                return
            self._start_operation(
                "preview", lambda: self.model.prepare_preview(adapter)
            )

        @Slot()
        def assemble_and_save(self) -> None:
            if self.job_is_running():
                self.status_changed.emit("A feature operation is already running.")
                return
            try:
                self._pull_values()
                adapter = coerce_feature_workflow(self._service)
                self.model.build_request(adapter)
            except Exception as exc:
                self._show_error(str(exc))
                return
            self._start_operation("build", lambda: self.model.assemble(adapter))

        def _set_busy(self, busy: bool) -> None:
            self.form_content.setEnabled(not busy)
            self.scan_button.setEnabled(not busy)
            self.input_preview_button.setEnabled(not busy)
            self.preview_button.setEnabled(not busy)
            self.build_button.setEnabled(not busy)

        def _start_operation(
            self, kind: str, operation: Callable[[], Any]
        ) -> None:
            if self.job_is_running():
                raise RuntimeError("A feature operation is already running.")
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
            self._set_busy(True)
            status = {
                "discover": "Reading placement CSV schemas…",
                "input_preview": (
                    "Loading body geometry and placement locations for visual preview…"
                ),
                "preview": "Validating placements and preparing preview…",
                "build": "Assembling coherent feature responses…",
            }[kind]
            self.status_changed.emit(status)
            thread.start()

        @Slot(object)
        def _operation_succeeded(self, result: Any) -> None:
            kind = self._active_kind
            if kind == "discover":
                current_paths = (
                    self.point_csv_picker.path(),
                    self.line_csv_picker.path(),
                )
                if current_paths != self._discovery_paths:
                    self.model.invalidate_dataset_requirements()
                    self._apply_requirements_to_tables()
                    self.status_changed.emit(
                        "CSV paths changed during discovery; re-scan them."
                    )
                    return
                self.model.update_dataset_requirements(result)
                self._apply_requirements_to_tables()
                point_count = len(self.model.point_dataset_ids)
                line_count = len(self.model.line_dataset_ids)
                missing = self.model.missing_dataset_mappings()
                if missing:
                    self.status_changed.emit(
                        f"✓ CSV schema valid: found {point_count} point and "
                        f"{line_count} line dataset ID(s). Next, choose an "
                        "OPN-FRD .grim response for: " + ", ".join(missing)
                    )
                else:
                    self.status_changed.emit(
                        f"✓ CSV schema valid: found {point_count} point and "
                        f"{line_count} line dataset ID(s); every response is mapped. "
                        "Next, validate placements and preview in 3-D."
                    )
            elif kind == "input_preview":
                requirements = getattr(result, "dataset_requirements", None)
                if requirements is not None:
                    self.model.update_dataset_requirements(requirements)
                    self._apply_requirements_to_tables()
                point_groups = getattr(result, "point_locations_cad_m", {})
                line_groups = getattr(result, "line_paths_cad_m", {})
                try:
                    point_total = sum(len(group) for group in point_groups.values())
                    line_total = sum(len(group) for group in line_groups.values())
                    count_text = (
                        f" ({point_total} point placement(s), "
                        f"{line_total} line path(s))"
                    )
                except (AttributeError, TypeError):
                    count_text = ""
                self._preview_is_current = True
                self.status_changed.emit(
                    "Input preview prepared"
                    + count_text
                    + ". Visual QA only: physical placement and response checks "
                    "have not run. Use the Assembly tree checkboxes to show or "
                    "hide the body and feature groups."
                )
                self.preview_ready.emit(result)
            elif kind == "preview":
                self._preview_is_current = True
                self.status_changed.emit(
                    "Placements validated and every dataset ID is mapped. "
                    "Showing the body and feature groups in the 3-D Assembly "
                    "view; full response compatibility is checked during "
                    "assembly, and tree checkboxes control display only."
                )
                self.preview_ready.emit(result)
            elif kind == "build":
                dispatch = result
                self._preview_is_current = True
                self.preview_ready.emit(dispatch.plan)
                self.feature_built.emit(str(dispatch.output_path))
                self.status_changed.emit(f"Saved assembled response: {dispatch.output_path}")

        @Slot(str)
        def _operation_failed(self, text: str) -> None:
            if self._active_kind == "discover":
                self.model.invalidate_dataset_requirements()
                self._apply_requirements_to_tables()
            self._show_error(text)

        @Slot()
        def _operation_thread_finished(self) -> None:
            self._thread = None
            self._worker = None
            self._active_kind = ""
            self._set_busy(False)


else:

    class FeatureAssemblyPanel:  # pragma: no cover - exercised only without Qt
        """Placeholder that preserves an actionable import-time API."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "FeatureAssemblyPanel requires PySide6. "
                f"Original import error: {_GUI_IMPORT_ERROR}"
            )


__all__ = [
    "GUI_AVAILABLE",
    "LINE_PLACEMENT_COLUMNS",
    "LINE_PLACEMENT_EXAMPLE",
    "POINT_PLACEMENT_COLUMNS",
    "POINT_PLACEMENT_EXAMPLE",
    "UNIT_CHOICES",
    "FeatureAssemblyFormModel",
    "FeatureAssemblyPanel",
    "FeatureAssemblyValues",
    "FeatureBuildDispatch",
    "FeatureWorkflowAdapter",
    "coerce_feature_workflow",
    "placement_csv_template_text",
    "write_placement_csv_template",
]
