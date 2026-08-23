"""Compact, non-blocking controls for coherent feature assembly.

The physics and the placement CSV schemas deliberately do not live here.
``FeatureWorkflowAdapter`` accepts the authoritative GHOST
``feature_workflow`` module (or a compatible injected service), while
``FeatureAssemblyFormModel`` keeps request construction testable without Qt or
GHOST on the import path.

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
        return cls(
            request_factory=module.FeatureAssemblyRequest,
            discover=module.discover_feature_dataset_ids,
            prepare=module.prepare_feature_assembly,
            execute=module.execute_feature_assembly,
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
        self.values.point_datasets = {
            dataset_id: _clean_path(self.values.point_datasets.get(dataset_id))
            for dataset_id in point_ids
        }
        self.values.line_datasets = {
            dataset_id: _clean_path(self.values.line_datasets.get(dataset_id))
            for dataset_id in line_ids
        }

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
                ["dataset_id", "OPN-FRD response (.grim)", ""]
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
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.empty_label)
            layout.addWidget(self.table)
            self.table.cellChanged.connect(lambda *_: self.mapping_changed.emit())
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
            self._build_ui()

        def _build_ui(self) -> None:
            outer = QVBoxLayout(self)
            outer.setContentsMargins(6, 6, 6, 6)
            outer.setSpacing(6)

            intro = QLabel(
                "Place compact or line features on a solved body. Feature "
                "responses must be coherent OPN-FRD datasets. Hiding an item "
                "in the 3-D preview does not exclude it from the build.",
                self,
            )
            intro.setWordWrap(True)
            outer.addWidget(intro)

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
            body_form.addRow("Base GRIM:", self.base_picker)
            body_form.addRow("STL/facet (optional for BoR):", self.surface_picker)
            body_form.addRow("Placement CSV units:", self.coordinate_units)
            body_form.addRow("Surface mesh units:", self.surface_units)
            body_form.addRow("Mesh options:", mesh_options)
            body_form.addRow("Output GRIM:", self.output_picker)
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
            point_layout.addWidget(QLabel("Point placement CSV:"))
            point_layout.addWidget(self.point_csv_picker)
            point_help = QLabel(
                "Each mapped OPN-FRD response must contain VV, HH, and "
                "reciprocal VH.",
                point_page,
            )
            point_help.setWordWrap(True)
            point_layout.addWidget(point_help)
            self.point_mapping = _DatasetMappingEditor(
                "Choose a point CSV to discover its dataset_id values.", point_page
            )
            point_layout.addWidget(self.point_mapping)
            self.feature_tabs.addTab(point_page, "Point features")

            line_page = QWidget(self.feature_tabs)
            line_layout = QVBoxLayout(line_page)
            self.line_csv_picker = _PathPicker(
                caption="Choose line placement CSV",
                file_filter="CSV placement file (*.csv);;All files (*)",
            )
            line_layout.addWidget(QLabel("Line placement CSV:"))
            line_layout.addWidget(self.line_csv_picker)
            line_help = QLabel(
                "Each mapped OPN-FRD line response must contain TE and TM.",
                line_page,
            )
            line_help.setWordWrap(True)
            line_layout.addWidget(line_help)
            self.line_mapping = _DatasetMappingEditor(
                "Choose a line CSV to discover its dataset_id values.", line_page
            )
            line_layout.addWidget(self.line_mapping)
            self.feature_tabs.addTab(line_page, "Line features")
            feature_layout.addWidget(self.feature_tabs)
            self.scan_button = QPushButton("Re-scan CSV dataset IDs", feature_group)
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

            action_row = QHBoxLayout()
            self.preview_button = QPushButton("Validate && Preview", self)
            self.build_button = QPushButton("Assemble && Save", self)
            self.build_button.setDefault(True)
            action_row.addStretch(1)
            action_row.addWidget(self.preview_button)
            action_row.addWidget(self.build_button)
            outer.addLayout(action_row)

            self.base_picker.editing_finished.connect(self._base_path_changed)
            self.point_csv_picker.editing_finished.connect(self.refresh_dataset_ids)
            self.line_csv_picker.editing_finished.connect(self.refresh_dataset_ids)
            self.scan_button.clicked.connect(self.refresh_dataset_ids)
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

        def set_point_csv(self, path: str, *, discover: bool = True) -> None:
            self.point_csv_picker.set_path(path)
            if discover:
                self.refresh_dataset_ids()

        def set_line_csv(self, path: str, *, discover: bool = True) -> None:
            self.line_csv_picker.set_path(path)
            if discover:
                self.refresh_dataset_ids()

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
            base = self.base_picker.path()
            if base and not self.output_picker.path():
                source = Path(base)
                suggestion = source.with_name(source.stem + "_features.grim")
                self.output_picker.set_path(str(suggestion))

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
                    self.status_changed.emit(
                        "CSV paths changed during discovery; re-scan them."
                    )
                    return
                self.model.update_dataset_requirements(result)
                self._apply_requirements_to_tables()
                point_count = len(self.model.point_dataset_ids)
                line_count = len(self.model.line_dataset_ids)
                self.status_changed.emit(
                    f"Found {point_count} point and {line_count} line dataset ID(s)."
                )
            elif kind == "preview":
                self.preview_ready.emit(result)
                self.status_changed.emit("Feature placements validated; preview updated.")
            elif kind == "build":
                dispatch = result
                self.preview_ready.emit(dispatch.plan)
                self.feature_built.emit(str(dispatch.output_path))
                self.status_changed.emit(f"Saved assembled response: {dispatch.output_path}")

        @Slot(str)
        def _operation_failed(self, text: str) -> None:
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
    "UNIT_CHOICES",
    "FeatureAssemblyFormModel",
    "FeatureAssemblyPanel",
    "FeatureAssemblyValues",
    "FeatureBuildDispatch",
    "FeatureWorkflowAdapter",
    "coerce_feature_workflow",
]
