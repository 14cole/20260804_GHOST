"""PowerPoint report workspace for the unified GRIM GUI.

The widget in this module owns presentation choices and slide preview only.
It delegates RCS extraction to :mod:`ppt_plot_data` and deterministic slide
planning/rendering/export to :mod:`ppt_report`.  Keeping those boundaries
separate prevents the GUI from introducing a second interpretation of the
underlying electromagnetic data.
"""

from __future__ import annotations

import math
import tempfile
import weakref
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ppt_plot_data import (
    NamedGrid,
    build_azimuth_specs,
    build_frequency_spec,
    get_plot_availability,
)
from ppt_report import (
    PlotSpec,
    PresentationPlan,
    SLIDE_FOOTER_FONT_SIZE_POINTS,
    SLIDE_PAGE_NUMBER_FONT_SIZE_POINTS,
    SLIDE_TITLE_FONT_SIZE_POINTS,
    SlidePlan,
    export_powerpoint_report,
    geometry_for_layout,
    plan_azimuth_slides,
    plan_frequency_slides,
    render_plot_png,
)


_INITIAL_AZIMUTH_FREQUENCY_COUNT = 6
_MAX_AZIMUTH_REPORT_FREQUENCIES = 60


@dataclass(frozen=True)
class DatasetCatalogEntry:
    """One stable GRIM dataset reference shown in the PPT catalog.

    ``dataset_id`` must remain unchanged when a row is renamed.  The main GUI
    can normally use its own row UUID; ``id(grid)`` is also stable for the
    lifetime of an in-memory dataset.
    """

    dataset_id: str
    name: str
    grid: Any
    source: str = ""

    def __post_init__(self) -> None:
        if not str(self.dataset_id).strip():
            raise ValueError("A PPT dataset entry requires a stable dataset_id.")
        if not str(self.name).strip():
            raise ValueError("A PPT dataset entry requires a display name.")
        if self.grid is None:
            raise ValueError("A PPT dataset entry requires an RcsGrid object.")


def _coerce_catalog_entry(value: Any) -> DatasetCatalogEntry:
    """Accept the public dataclass plus convenient shell-facing shapes."""

    if isinstance(value, DatasetCatalogEntry):
        return value
    if isinstance(value, Mapping):
        grid = value.get("grid", value.get("dataset"))
        dataset_id = value.get("dataset_id", value.get("id", value.get("key")))
        name = value.get("name", value.get("label"))
        source = value.get("source", value.get("path", ""))
        return DatasetCatalogEntry(str(dataset_id or ""), str(name or ""), grid, str(source or ""))
    if isinstance(value, (tuple, list)):
        if len(value) == 3:
            dataset_id, name, grid = value
            return DatasetCatalogEntry(str(dataset_id), str(name), grid)
        if len(value) == 4:
            dataset_id, name, grid, source = value
            return DatasetCatalogEntry(
                str(dataset_id), str(name), grid, str(source or "")
            )
    dataset_id = getattr(value, "dataset_id", getattr(value, "id", None))
    name = getattr(value, "name", getattr(value, "label", None))
    grid = getattr(value, "grid", getattr(value, "dataset", None))
    source = getattr(value, "source", getattr(value, "path", ""))
    if dataset_id is not None or name is not None or grid is not None:
        return DatasetCatalogEntry(
            str(dataset_id or ""), str(name or ""), grid, str(source or "")
        )
    raise TypeError(
        "PPT dataset entries must be DatasetCatalogEntry values, mappings, "
        "(dataset_id, name, grid) tuples, or objects with matching attributes."
    )


def _finite_common_y_limits(plots: Sequence[PlotSpec]) -> tuple[float, float]:
    """Return one readable rounded vertical scale shared by every plot."""

    low = math.inf
    high = -math.inf
    for plot in plots:
        for series in plot.series:
            for value in series.y:
                number = float(value)
                if math.isfinite(number):
                    low = min(low, number)
                    high = max(high, number)
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("The selected cuts contain no finite magnitude samples.")
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1.0e-12):
        low -= 5.0
        high += 5.0
    else:
        padding = max(1.0, 0.05 * (high - low))
        low -= padding
        high += padding
    step = 5.0 if high - low <= 80.0 else 10.0
    rounded_low = step * math.floor(low / step)
    rounded_high = step * math.ceil(high / step)
    if rounded_high <= rounded_low:
        rounded_high = rounded_low + step
    return float(rounded_low), float(rounded_high)


def _with_shared_y_limits(
    plots: Sequence[PlotSpec], limits: tuple[float, float] | None = None
) -> tuple[PlotSpec, ...]:
    """Copy plot specs with a single fixed scale so slides do not jump."""

    values = tuple(plots)
    shared = limits if limits is not None else _finite_common_y_limits(values)
    return tuple(replace(plot, y_limits=shared) for plot in values)


_GUI_IMPORT_ERROR: Exception | None = None
try:  # Keep report planning importable on headless/minimal installations.
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.image import imread
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
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )
except (ImportError, RuntimeError) as exc:  # pragma: no cover - environment-specific
    _GUI_IMPORT_ERROR = exc


GUI_AVAILABLE = _GUI_IMPORT_ERROR is None


if GUI_AVAILABLE:

    _CATALOG_ID_ROLE = Qt.ItemDataRole.UserRole + 31
    _AXIS_VALUE_ROLE = Qt.ItemDataRole.UserRole + 32


    class _ExportWorker(QObject):
        succeeded = Signal(str)
        failed = Signal(str)

        def __init__(
            self,
            exporter: Callable[..., Any],
            plan: PresentationPlan,
            output_path: str,
            template_path: str,
        ) -> None:
            super().__init__()
            self._exporter = exporter
            self._plan = plan
            self._output_path = output_path
            self._template_path = template_path

        @Slot()
        def run(self) -> None:
            try:
                result = self._exporter(
                    self._plan,
                    self._output_path,
                    template_path=self._template_path or None,
                )
            except Exception as exc:
                self.failed.emit(str(exc).strip() or type(exc).__name__)
            else:
                self.succeeded.emit(str(result or self._output_path))


    class _SlidePreviewCanvas(FigureCanvas):
        """White 16:9 slide canvas driven by the exact export geometry."""

        def __init__(self, parent: QWidget | None = None) -> None:
            self.figure = Figure(figsize=(13.3333333333, 7.5), dpi=90)
            self.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
            super().__init__(self.figure)
            self.setParent(parent)
            self.setMinimumSize(520, 293)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.show_feedback(
                "Slide preview",
                "Choose datasets and plot settings, then select Build Preview.",
            )

        def _blank_slide(self) -> Any:
            self.figure.clear()
            self.figure.patch.set_facecolor("#e7edf5")
            slide = self.figure.add_axes((0.018, 0.018, 0.964, 0.964))
            slide.set_facecolor("#ffffff")
            slide.set_xticks(())
            slide.set_yticks(())
            for spine in slide.spines.values():
                spine.set_color("#9aa9bc")
                spine.set_linewidth(1.1)
            slide.set_xlim(0.0, 1.0)
            slide.set_ylim(0.0, 1.0)
            return slide

        def show_feedback(self, heading: str, detail: str = "") -> None:
            slide = self._blank_slide()
            slide.text(
                0.5,
                0.54,
                str(heading),
                ha="center",
                va="center",
                color="#172033",
                fontsize=18,
                fontweight="bold",
                family="Arial",
            )
            if detail:
                slide.text(
                    0.5,
                    0.45,
                    str(detail),
                    ha="center",
                    va="top",
                    color="#59687c",
                    fontsize=10,
                    family="Arial",
                    wrap=True,
                )
            self.draw_idle()

        def render_slide(
            self,
            slide_plan: SlidePlan,
            *,
            slide_index: int,
            slide_count: int,
            output_directory: Path,
            generation: int,
        ) -> None:
            """Render one page through ``render_plot_png`` then place its images."""

            self._blank_slide()
            geometry = geometry_for_layout(slide_plan.layout)
            # The actual white page occupies 96.4% of the figure with a small
            # application-background border around it.
            page_left, page_bottom, page_width, page_height = 0.018, 0.018, 0.964, 0.964

            def x_position(value: float) -> float:
                return page_left + page_width * value / geometry.width

            def y_position_from_top(value: float) -> float:
                return page_bottom + page_height * (1.0 - value / geometry.height)

            title_center_y = slide_plan_title_y = (
                geometry.title.top + 0.52 * geometry.title.height
            )
            self.figure.text(
                x_position(geometry.title.left),
                y_position_from_top(title_center_y),
                slide_plan.title,
                ha="left",
                va="center",
                color="#172033",
                fontsize=SLIDE_TITLE_FONT_SIZE_POINTS,
                fontweight="bold",
                family="Arial",
            )
            output_directory.mkdir(parents=True, exist_ok=True)
            for placement_index, placement in enumerate(slide_plan.plots):
                image_path = output_directory / (
                    f"preview_{generation:04d}_slide_{slide_index + 1:03d}_"
                    f"slot_{placement.slot_index + 1}.png"
                )
                if not image_path.is_file():
                    render_plot_png(
                        placement.plot,
                        image_path,
                        width_points=placement.frame.width,
                        height_points=placement.frame.height,
                        dpi=120,
                    )
                frame = placement.frame
                axes = self.figure.add_axes(
                    (
                        x_position(frame.left),
                        y_position_from_top(frame.bottom),
                        page_width * frame.width / geometry.width,
                        page_height * frame.height / geometry.height,
                    )
                )
                axes.imshow(imread(image_path), aspect="auto")
                axes.set_axis_off()
            if slide_plan.footer:
                self.figure.text(
                    x_position(geometry.footer.left),
                    y_position_from_top(
                        geometry.footer.top + 0.58 * geometry.footer.height
                    ),
                    slide_plan.footer,
                    ha="left",
                    va="center",
                    color="#48566a",
                    fontsize=SLIDE_FOOTER_FONT_SIZE_POINTS,
                    family="Arial",
                )
            self.figure.text(
                x_position(geometry.page_number.right),
                y_position_from_top(
                    geometry.page_number.top + 0.58 * geometry.page_number.height
                ),
                f"{slide_index + 1} / {slide_count}",
                ha="right",
                va="center",
                color="#48566a",
                fontsize=SLIDE_PAGE_NUMBER_FONT_SIZE_POINTS,
                family="Arial",
            )
            self.draw_idle()


    class PptWorkspace(QWidget):
        """Top-level GRIM workspace for uniform, previewed PPTX reports."""

        report_exported = Signal(str)
        status_changed = Signal(str)
        main_selection_requested = Signal()

        def __init__(
            self,
            parent: QWidget | None = None,
            *,
            exporter: Callable[..., Any] = export_powerpoint_report,
            selected_ids_provider: Callable[[], Iterable[str]] | None = None,
        ) -> None:
            super().__init__(parent)
            self._exporter = exporter
            self._selected_ids_provider = selected_ids_provider
            self._catalog: dict[str, DatasetCatalogEntry] = {}
            self._availability: Any = None
            self._syncing = False
            self._frequency_choices_initialized = False
            self._preview_plan: PresentationPlan | None = None
            self._preview_is_current = False
            self._preview_generation = 0
            self._current_slide_index = 0
            self._thread: QThread | None = None
            self._worker: _ExportWorker | None = None
            self._last_error = ""
            self._preview_temp = tempfile.TemporaryDirectory(prefix="grim-ppt-preview-")
            # Embedded hosts normally call dispose(), but a Python/Qt wrapper
            # can outlive its native widget during teardown or abnormal
            # interpreter shutdown. This later finalizer removes preview files
            # explicitly before TemporaryDirectory needs to warn about them.
            self._preview_finalizer = weakref.finalize(
                self, self._preview_temp.cleanup
            )
            self._disposed = False
            self._build_ui()
            self._connect_signals()
            self._update_plot_type_controls()
            self.fixed_scale_widget.setVisible(False)
            self._refresh_availability()

        # ------------------------------------------------------------------
        # Public shell integration contract
        # ------------------------------------------------------------------
        @property
        def preview_plan(self) -> PresentationPlan | None:
            return self._preview_plan

        @property
        def preview_is_current(self) -> bool:
            return self._preview_is_current

        @property
        def current_slide_index(self) -> int:
            return self._current_slide_index

        @property
        def last_error(self) -> str:
            return self._last_error

        def set_dataset_catalog(self, entries: Iterable[Any]) -> None:
            """Replace the available rows while preserving ID state and order."""

            values = tuple(_coerce_catalog_entry(entry) for entry in entries)
            incoming = {entry.dataset_id: entry for entry in values}
            if len(incoming) != len(values):
                raise ValueError("PPT dataset_id values must be unique.")
            old_order = self.dataset_ids_in_order()
            old_checked = set(self.selected_dataset_ids())
            had_catalog = bool(self._catalog)
            order = [dataset_id for dataset_id in old_order if dataset_id in incoming]
            order.extend(
                entry.dataset_id for entry in values if entry.dataset_id not in order
            )
            self._catalog = incoming
            self._syncing = True
            self.dataset_list.blockSignals(True)
            self.dataset_list.clear()
            for dataset_id in order:
                entry = incoming[dataset_id]
                item = QListWidgetItem(entry.name)
                item.setData(_CATALOG_ID_ROLE, dataset_id)
                item.setFlags(
                    item.flags()
                    | Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsDragEnabled
                    | Qt.ItemFlag.ItemIsDropEnabled
                )
                checked = dataset_id in old_checked if had_catalog else True
                if had_catalog and dataset_id not in old_order:
                    checked = True
                item.setCheckState(
                    Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                )
                source_text = entry.source or str(
                    getattr(entry.grid, "source_path", "") or ""
                )
                item.setToolTip(
                    f"{entry.name}\n{source_text}" if source_text else entry.name
                )
                self.dataset_list.addItem(item)
            self.dataset_list.blockSignals(False)
            self._syncing = False
            self._dataset_selection_changed()

        def dataset_ids_in_order(self) -> tuple[str, ...]:
            return tuple(
                str(self.dataset_list.item(index).data(_CATALOG_ID_ROLE))
                for index in range(self.dataset_list.count())
            )

        def selected_dataset_ids(self) -> tuple[str, ...]:
            return tuple(
                str(item.data(_CATALOG_ID_ROLE))
                for item in (
                    self.dataset_list.item(index)
                    for index in range(self.dataset_list.count())
                )
                if item.checkState() == Qt.CheckState.Checked
            )

        def select_dataset_ids(self, dataset_ids: Iterable[str]) -> None:
            wanted = {str(value) for value in dataset_ids}
            self._syncing = True
            self.dataset_list.blockSignals(True)
            for index in range(self.dataset_list.count()):
                item = self.dataset_list.item(index)
                item.setCheckState(
                    Qt.CheckState.Checked
                    if str(item.data(_CATALOG_ID_ROLE)) in wanted
                    else Qt.CheckState.Unchecked
                )
            self.dataset_list.blockSignals(False)
            self._syncing = False
            self._dataset_selection_changed()

        set_selected_dataset_ids = select_dataset_ids

        def set_selected_ids_provider(
            self, provider: Callable[[], Iterable[str]] | None
        ) -> None:
            self._selected_ids_provider = provider

        def selected_frequencies(self) -> tuple[Any, ...]:
            return tuple(
                item.data(_AXIS_VALUE_ROLE)
                for item in (
                    self.frequency_list.item(index)
                    for index in range(self.frequency_list.count())
                )
                if item.checkState() == Qt.CheckState.Checked
            )

        def select_frequencies(self, frequencies: Iterable[Any]) -> None:
            wanted = tuple(frequencies)
            self._syncing = True
            self.frequency_list.blockSignals(True)
            for index in range(self.frequency_list.count()):
                item = self.frequency_list.item(index)
                raw = item.data(_AXIS_VALUE_ROLE)
                checked = any(_axis_values_equal(raw, target) for target in wanted)
                item.setCheckState(
                    Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                )
            self.frequency_list.blockSignals(False)
            self._syncing = False
            self._frequency_choices_initialized = True
            self._mark_preview_stale()

        def set_plot_kind(self, kind: str) -> None:
            index = self.plot_type_combo.findData(str(kind))
            if index < 0:
                raise ValueError(f"Unknown PPT plot kind: {kind!r}")
            self.plot_type_combo.setCurrentIndex(index)

        def job_is_running(self) -> bool:
            return bool(self._thread is not None and self._thread.isRunning())

        def busy_operation(self) -> str | None:
            return "PowerPoint report export" if self.job_is_running() else None

        def focus_workspace(self) -> None:
            self.build_preview_button.setFocus(Qt.FocusReason.OtherFocusReason)

        # ------------------------------------------------------------------
        # UI construction
        # ------------------------------------------------------------------
        def _build_ui(self) -> None:
            outer = QVBoxLayout(self)
            outer.setContentsMargins(8, 8, 8, 8)
            outer.setSpacing(6)
            heading = QLabel("PowerPoint report builder", self)
            heading.setStyleSheet("font-size: 15px; font-weight: 600;")
            intro = QLabel(
                "Choose loaded GRIM datasets and build a slide preview before "
                "export. Fixed report layouts keep plots aligned from slide to "
                "slide and across analysts.",
                self,
            )
            intro.setWordWrap(True)
            outer.addWidget(heading)
            outer.addWidget(intro)

            splitter = QSplitter(Qt.Orientation.Horizontal, self)
            splitter.setChildrenCollapsible(False)
            outer.addWidget(splitter, 1)

            self.controls_scroll = QScrollArea(splitter)
            self.controls_scroll.setObjectName("pptControlsScroll")
            self.controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
            self.controls_scroll.setWidgetResizable(True)
            self.controls_scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self.controls_scroll.setMinimumWidth(405)
            self.controls_scroll.setMaximumWidth(485)
            self.controls_content = QWidget(self.controls_scroll)
            self.controls_content.setObjectName("pptControlsContent")
            controls = QVBoxLayout(self.controls_content)
            controls.setContentsMargins(2, 2, 8, 2)
            controls.setSpacing(8)

            dataset_group = QGroupBox("1  Datasets and plot order", self.controls_content)
            dataset_layout = QVBoxLayout(dataset_group)
            dataset_help = QLabel(
                "Checked datasets are overlaid in the order shown. Drag rows "
                "to set legend and line order; this list is independent of the "
                "main Plotting table.",
                dataset_group,
            )
            dataset_help.setWordWrap(True)
            dataset_layout.addWidget(dataset_help)
            self.dataset_list = QListWidget(dataset_group)
            self.dataset_list.setObjectName("pptDatasetCatalog")
            self.dataset_list.setMinimumHeight(120)
            self.dataset_list.setSelectionMode(
                QAbstractItemView.SelectionMode.ExtendedSelection
            )
            self.dataset_list.setDragDropMode(
                QAbstractItemView.DragDropMode.InternalMove
            )
            self.dataset_list.setDefaultDropAction(Qt.DropAction.MoveAction)
            self.dataset_list.setAlternatingRowColors(True)
            dataset_layout.addWidget(self.dataset_list)
            dataset_buttons = QHBoxLayout()
            self.use_selected_button = QPushButton("Use main selection", dataset_group)
            self.use_selected_button.setToolTip(
                "Check exactly the datasets selected in GRIM's main Plotting table."
            )
            self.all_datasets_button = QPushButton("All", dataset_group)
            self.no_datasets_button = QPushButton("None", dataset_group)
            dataset_buttons.addWidget(self.use_selected_button)
            dataset_buttons.addStretch(1)
            dataset_buttons.addWidget(self.all_datasets_button)
            dataset_buttons.addWidget(self.no_datasets_button)
            dataset_layout.addLayout(dataset_buttons)
            self.dataset_summary_label = QLabel("No datasets loaded.", dataset_group)
            self.dataset_summary_label.setWordWrap(True)
            dataset_layout.addWidget(self.dataset_summary_label)
            controls.addWidget(dataset_group)

            plot_group = QGroupBox("2  Plot definition", self.controls_content)
            plot_layout = QVBoxLayout(plot_group)
            plot_form = QFormLayout()
            plot_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            self.plot_type_combo = QComboBox(plot_group)
            self.plot_type_combo.addItem("Azimuth — rectangular", "azimuth_rect")
            self.plot_type_combo.addItem("Azimuth — polar", "azimuth_polar")
            self.plot_type_combo.addItem("Frequency sweep", "frequency")
            plot_form.addRow("Plot type", self.plot_type_combo)
            self.elevation_label = QLabel("Elevation cut", plot_group)
            self.elevation_combo = QComboBox(plot_group)
            plot_form.addRow(self.elevation_label, self.elevation_combo)
            self.azimuth_label = QLabel("Azimuth cut", plot_group)
            self.azimuth_combo = QComboBox(plot_group)
            plot_form.addRow(self.azimuth_label, self.azimuth_combo)
            self.polarization_combo = QComboBox(plot_group)
            plot_form.addRow("Polarization", self.polarization_combo)
            self.quantity_label = QLabel("Magnitude (dB)", plot_group)
            self.quantity_label.setToolTip(
                "The initial PPT workflow exports calibrated RCS magnitude. "
                "No phase convention is altered by this tab."
            )
            plot_form.addRow("Quantity", self.quantity_label)
            plot_layout.addLayout(plot_form)

            self.frequency_box = QWidget(plot_group)
            frequency_box_layout = QVBoxLayout(self.frequency_box)
            frequency_box_layout.setContentsMargins(0, 2, 0, 0)
            frequency_heading = QHBoxLayout()
            frequency_heading.addWidget(
                QLabel(
                    "Frequencies (one plot each; maximum 60 per report)",
                    self.frequency_box,
                )
            )
            frequency_heading.addStretch(1)
            self.all_frequencies_button = QPushButton("First 60", self.frequency_box)
            self.all_frequencies_button.setToolTip(
                "Select up to the first 60 common frequencies (10 slides). "
                "Create additional reports for larger frequency sets."
            )
            self.no_frequencies_button = QPushButton("None", self.frequency_box)
            frequency_heading.addWidget(self.all_frequencies_button)
            frequency_heading.addWidget(self.no_frequencies_button)
            frequency_box_layout.addLayout(frequency_heading)
            self.frequency_list = QListWidget(self.frequency_box)
            self.frequency_list.setObjectName("pptFrequencyCatalog")
            self.frequency_list.setMinimumHeight(116)
            self.frequency_list.setAlternatingRowColors(True)
            frequency_box_layout.addWidget(self.frequency_list)
            plot_layout.addWidget(self.frequency_box)

            scale_form = QFormLayout()
            self.scale_mode_combo = QComboBox(plot_group)
            self.scale_mode_combo.addItem("Shared automatic", "shared_auto")
            self.scale_mode_combo.addItem("Fixed", "fixed")
            self.scale_mode_combo.setToolTip(
                "Every plot in the report uses the same vertical RCS scale."
            )
            scale_form.addRow("RCS scale", self.scale_mode_combo)
            self.fixed_scale_widget = QWidget(plot_group)
            fixed_scale_layout = QHBoxLayout(self.fixed_scale_widget)
            fixed_scale_layout.setContentsMargins(0, 0, 0, 0)
            self.y_min_spin = QDoubleSpinBox(self.fixed_scale_widget)
            self.y_min_spin.setRange(-500.0, 500.0)
            self.y_min_spin.setDecimals(1)
            self.y_min_spin.setValue(-60.0)
            self.y_max_spin = QDoubleSpinBox(self.fixed_scale_widget)
            self.y_max_spin.setRange(-500.0, 500.0)
            self.y_max_spin.setDecimals(1)
            self.y_max_spin.setValue(20.0)
            fixed_scale_layout.addWidget(QLabel("Min", self.fixed_scale_widget))
            fixed_scale_layout.addWidget(self.y_min_spin)
            fixed_scale_layout.addWidget(QLabel("Max", self.fixed_scale_widget))
            fixed_scale_layout.addWidget(self.y_max_spin)
            scale_form.addRow("Fixed range", self.fixed_scale_widget)
            self.show_legend_check = QCheckBox("Show dataset legend", plot_group)
            self.show_legend_check.setChecked(True)
            scale_form.addRow("", self.show_legend_check)
            plot_layout.addLayout(scale_form)
            self.layout_value_label = QLabel(plot_group)
            self.layout_value_label.setWordWrap(True)
            self.layout_value_label.setObjectName("pptFixedLayoutLabel")
            plot_layout.addWidget(self.layout_value_label)
            controls.addWidget(plot_group)

            deck_group = QGroupBox("3  Slide text and files", self.controls_content)
            deck_form = QFormLayout(deck_group)
            deck_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            self.deck_title_edit = QLineEdit("RCS Report", deck_group)
            self.deck_title_edit.setPlaceholderText("RCS Report")
            deck_form.addRow("Slide title", self.deck_title_edit)
            self.footer_edit = QLineEdit(deck_group)
            self.footer_edit.setPlaceholderText("Program | classification | analyst")
            deck_form.addRow("Footer", self.footer_edit)
            self.template_edit, template_widget = self._path_row(
                deck_group, "Choose blank PowerPoint template", self._browse_template
            )
            self.template_edit.setPlaceholderText("Optional blank .pptx or .potx")
            self.template_edit.setToolTip(
                "Optional blank widescreen 16:9 .pptx or .potx. GRIM clears "
                "template slides but retains the presentation theme and master graphics."
            )
            deck_form.addRow("Blank template", template_widget)
            template_note = QLabel(
                "The preview shows GRIM content on a white 16:9 page. Template theme "
                "and master graphics appear only in the exported PPTX, so review the "
                "finished deck when using a custom template.",
                deck_group,
            )
            template_note.setWordWrap(True)
            template_note.setObjectName("pptTemplatePreviewNote")
            deck_form.addRow("", template_note)
            self.output_edit, output_widget = self._path_row(
                deck_group, "Choose report output", self._browse_output
            )
            self.output_edit.setText("RCS_Report.pptx")
            deck_form.addRow("Output .pptx", output_widget)
            controls.addWidget(deck_group)

            action_row = QHBoxLayout()
            self.build_preview_button = QPushButton("Build Preview", self.controls_content)
            self.build_preview_button.setDefault(True)
            self.export_button = QPushButton("Export PPTX", self.controls_content)
            self.export_button.setEnabled(False)
            action_row.addWidget(self.build_preview_button)
            action_row.addWidget(self.export_button)
            controls.addLayout(action_row)
            controls.addStretch(1)
            self.controls_scroll.setWidget(self.controls_content)
            splitter.addWidget(self.controls_scroll)

            preview_panel = QWidget(splitter)
            preview_layout = QVBoxLayout(preview_panel)
            preview_layout.setContentsMargins(8, 0, 0, 0)
            preview_header = QHBoxLayout()
            preview_title = QLabel("Slide preview", preview_panel)
            preview_title.setStyleSheet("font-weight: 600;")
            self.previous_slide_button = QPushButton("‹ Previous", preview_panel)
            self.next_slide_button = QPushButton("Next ›", preview_panel)
            self.page_label = QLabel("No preview", preview_panel)
            self.page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            preview_header.addWidget(preview_title)
            preview_header.addStretch(1)
            preview_header.addWidget(self.previous_slide_button)
            preview_header.addWidget(self.page_label)
            preview_header.addWidget(self.next_slide_button)
            preview_layout.addLayout(preview_header)
            self.preview_canvas = _SlidePreviewCanvas(preview_panel)
            preview_layout.addWidget(self.preview_canvas, 1)
            self.status_label = QLabel(
                "No preview yet. Select datasets and click Build Preview.", preview_panel
            )
            self.status_label.setWordWrap(True)
            self.status_label.setObjectName("pptStatusLabel")
            preview_layout.addWidget(self.status_label)
            splitter.addWidget(preview_panel)
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes((430, 900))
            self._update_navigation()

        def _path_row(
            self,
            parent: QWidget,
            accessible_name: str,
            browse: Callable[[], None],
        ) -> tuple[QLineEdit, QWidget]:
            widget = QWidget(parent)
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            edit = QLineEdit(widget)
            edit.setAccessibleName(accessible_name)
            button = QPushButton("Browse…", widget)
            button.clicked.connect(browse)
            layout.addWidget(edit, 1)
            layout.addWidget(button)
            return edit, widget

        def _connect_signals(self) -> None:
            self.dataset_list.itemChanged.connect(self._dataset_selection_changed)
            self.dataset_list.model().rowsMoved.connect(self._dataset_selection_changed)
            self.use_selected_button.clicked.connect(self._use_main_selection)
            self.all_datasets_button.clicked.connect(
                lambda: self.select_dataset_ids(self.dataset_ids_in_order())
            )
            self.no_datasets_button.clicked.connect(lambda: self.select_dataset_ids(()))
            self.plot_type_combo.currentIndexChanged.connect(
                self._update_plot_type_controls
            )
            self.frequency_list.itemChanged.connect(self._mark_preview_stale)
            self.all_frequencies_button.clicked.connect(self._select_first_frequencies)
            self.no_frequencies_button.clicked.connect(
                lambda: self.select_frequencies(())
            )
            for combo in (
                self.elevation_combo,
                self.azimuth_combo,
                self.polarization_combo,
                self.scale_mode_combo,
            ):
                combo.currentIndexChanged.connect(self._control_changed)
            for spin in (self.y_min_spin, self.y_max_spin):
                spin.valueChanged.connect(self._mark_preview_stale)
            self.show_legend_check.toggled.connect(self._mark_preview_stale)
            self.deck_title_edit.textChanged.connect(self._mark_preview_stale)
            self.footer_edit.textChanged.connect(self._mark_preview_stale)
            self.template_edit.textChanged.connect(self._mark_preview_stale)
            self.build_preview_button.clicked.connect(self.build_preview)
            self.export_button.clicked.connect(self.export_report)
            self.previous_slide_button.clicked.connect(self.previous_slide)
            self.next_slide_button.clicked.connect(self.next_slide)

        # ------------------------------------------------------------------
        # Selector synchronization and validation
        # ------------------------------------------------------------------
        @Slot()
        def _use_main_selection(self) -> None:
            if self._selected_ids_provider is None:
                self.main_selection_requested.emit()
                self._set_status(
                    "Waiting for the main Plotting table selection. If this "
                    "button is not connected by the shell, check rows here directly."
                )
                return
            try:
                self.select_dataset_ids(self._selected_ids_provider())
            except Exception as exc:
                self._show_error(str(exc))

        @Slot()
        def _dataset_selection_changed(self, *_args: Any) -> None:
            if self._syncing:
                return
            self._refresh_availability()
            self._mark_preview_stale()

        def _selected_entries(self) -> tuple[DatasetCatalogEntry, ...]:
            return tuple(
                self._catalog[dataset_id]
                for dataset_id in self.selected_dataset_ids()
                if dataset_id in self._catalog
            )

        def _named_grids(self) -> tuple[NamedGrid, ...]:
            return tuple(
                NamedGrid(entry.name, entry.grid) for entry in self._selected_entries()
            )

        def _refresh_availability(self) -> None:
            selected = self._selected_entries()
            total = len(self._catalog)
            self.dataset_summary_label.setText(
                f"{len(selected)} of {total} loaded dataset(s) selected."
                if total
                else "No datasets loaded. Drop or open data in GRIM first."
            )
            if not selected:
                self._availability = None
                self._frequency_choices_initialized = False
                self.azimuth_label.setText("Azimuth cut")
                self.elevation_label.setText("Elevation cut")
                self._set_axis_combo(self.elevation_combo, (), "")
                self._set_axis_combo(self.azimuth_combo, (), "")
                self._set_axis_combo(self.polarization_combo, (), "")
                self._set_frequency_choices((), "")
                return
            try:
                availability = get_plot_availability(
                    self._named_grids(),
                    evaluate_phase=False,
                )
            except Exception as exc:
                self._availability = None
                self.azimuth_label.setText("Azimuth cut")
                self.elevation_label.setText("Elevation cut")
                self.dataset_summary_label.setText(
                    f"Selected datasets are not plot-compatible: {exc}"
                )
                self._set_axis_combo(self.elevation_combo, (), "")
                self._set_axis_combo(self.azimuth_combo, (), "")
                self._set_axis_combo(self.polarization_combo, (), "")
                self._set_frequency_choices((), "")
                return
            self._availability = availability
            great_circle = (
                str(getattr(availability, "angular_coordinate_system", "conic"))
                == "great_circle"
            )
            self.azimuth_label.setText(
                "Aspect cut" if great_circle else "Azimuth cut"
            )
            self.elevation_label.setText(
                "Pitch cut" if great_circle else "Elevation cut"
            )
            angle_unit = str(
                getattr(
                    availability,
                    "angle_unit",
                    getattr(availability, "azimuth_unit", "deg"),
                )
            )
            elevation_unit = str(getattr(availability, "elevation_unit", angle_unit))
            frequency_unit = str(getattr(availability, "frequency_unit", "GHz"))
            self._set_axis_combo(
                self.elevation_combo,
                tuple(availability.elevations),
                elevation_unit,
                preferred=0.0,
            )
            self._set_axis_combo(
                self.azimuth_combo,
                tuple(availability.azimuths),
                angle_unit,
                preferred=0.0,
            )
            self._set_axis_combo(
                self.polarization_combo,
                tuple(availability.polarizations),
                "",
                preferred="HH",
            )
            self._set_frequency_choices(
                tuple(availability.frequencies), frequency_unit
            )

        def _set_axis_combo(
            self,
            combo: QComboBox,
            values: Sequence[Any],
            unit: str,
            *,
            preferred: Any = None,
        ) -> None:
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for value in values:
                combo.addItem(_format_axis_value(value, unit), _plain_scalar(value))
            target = current if current is not None else preferred
            target_index = _find_combo_value(combo, target)
            if target_index < 0 and combo.count():
                target_index = 0
            combo.setCurrentIndex(target_index)
            combo.blockSignals(False)

        def _set_frequency_choices(
            self, values: Sequence[Any], unit: str
        ) -> None:
            old_values = self.selected_frequencies()
            initialized = self._frequency_choices_initialized
            self.frequency_list.blockSignals(True)
            self.frequency_list.clear()
            for index, value in enumerate(values):
                raw = _plain_scalar(value)
                item = QListWidgetItem(_format_frequency_value(raw, unit))
                item.setData(_AXIS_VALUE_ROLE, raw)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                checked = (
                    any(_axis_values_equal(raw, old) for old in old_values)
                    if initialized
                    else index < _INITIAL_AZIMUTH_FREQUENCY_COUNT
                )
                item.setCheckState(
                    Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                )
                self.frequency_list.addItem(item)
            self.frequency_list.blockSignals(False)
            if values:
                self._frequency_choices_initialized = True

        @Slot()
        def _select_first_frequencies(self) -> None:
            count = min(self.frequency_list.count(), _MAX_AZIMUTH_REPORT_FREQUENCIES)
            self.select_frequencies(
                self.frequency_list.item(index).data(_AXIS_VALUE_ROLE)
                for index in range(count)
            )
            if self.frequency_list.count() > count:
                self._set_status(
                    f"Selected the first {count} frequencies. The PPT builder limits "
                    "one report to 60 frequencies (10 azimuth slides) to keep preview "
                    "and export responsive."
                )

        @Slot()
        def _update_plot_type_controls(self, *_args: Any) -> None:
            kind = str(self.plot_type_combo.currentData() or "azimuth_rect")
            azimuth_kind = kind in {"azimuth_rect", "azimuth_polar"}
            self.frequency_box.setVisible(azimuth_kind)
            self.azimuth_label.setVisible(not azimuth_kind)
            self.azimuth_combo.setVisible(not azimuth_kind)
            if azimuth_kind:
                name = "rectangular" if kind == "azimuth_rect" else "polar"
                self.layout_value_label.setText(
                    f"Fixed layout: six {name} azimuth plots per slide "
                    "(3 columns × 2 rows). Frequencies continue row-major on "
                    "additional slides. All plots share one RCS scale."
                )
            else:
                self.layout_value_label.setText(
                    "Fixed layout: one full-width frequency sweep per slide. "
                    "Selected datasets are overlaid and use one shared RCS scale."
                )
            self._mark_preview_stale()

        @Slot()
        def _control_changed(self, *_args: Any) -> None:
            self.fixed_scale_widget.setVisible(
                self.scale_mode_combo.currentData() == "fixed"
            )
            self._mark_preview_stale()

        @Slot()
        def _mark_preview_stale(self, *_args: Any) -> None:
            if self._syncing or not self._preview_is_current:
                return
            self._preview_is_current = False
            self.export_button.setEnabled(False)
            message = "Settings changed — click Build Preview to verify the updated slides."
            self.preview_canvas.show_feedback("Preview is out of date", message)
            self.page_label.setText("Stale preview")
            self._set_status(message)
            self._update_navigation()

        def _fixed_y_limits(self) -> tuple[float, float] | None:
            if self.scale_mode_combo.currentData() != "fixed":
                return None
            low = float(self.y_min_spin.value())
            high = float(self.y_max_spin.value())
            if low >= high:
                raise ValueError("Fixed RCS minimum must be less than the maximum.")
            return low, high

        def _build_plan(self) -> PresentationPlan:
            datasets = self._named_grids()
            if not datasets:
                raise ValueError(
                    "Select at least one loaded dataset in the PPT dataset list."
                )
            # Re-run compatibility here so preview always validates the current
            # catalog even if a grid object was replaced in place.
            availability = get_plot_availability(
                datasets,
                evaluate_phase=False,
            )
            kind = str(self.plot_type_combo.currentData())
            elevation = self.elevation_combo.currentData()
            polarization = self.polarization_combo.currentData()
            if elevation is None:
                raise ValueError("The selected datasets have no common elevation cut.")
            if polarization is None:
                raise ValueError("The selected datasets have no common polarization.")
            fixed_limits = self._fixed_y_limits()
            show_legend = self.show_legend_check.isChecked()
            deck_title = self.deck_title_edit.text().strip()
            footer = self.footer_edit.text().strip()
            if kind in {"azimuth_rect", "azimuth_polar"}:
                frequencies = self.selected_frequencies()
                if not frequencies:
                    raise ValueError(
                        "Select one or more frequencies for the azimuth report."
                    )
                if len(frequencies) > _MAX_AZIMUTH_REPORT_FREQUENCIES:
                    raise ValueError(
                        "A single PPT report is limited to 60 azimuth frequencies "
                        "(10 slides) to keep preview and export responsive. Select a "
                        "smaller frequency group and create additional reports as needed."
                    )
                if kind == "azimuth_polar" and not bool(
                    getattr(availability, "polar_available", True)
                ):
                    raise ValueError(
                        "Polar plotting is unavailable for the selected azimuth axis. "
                        "Use rectangular azimuth or choose datasets with compatible "
                        "angular coverage."
                    )
                plots = build_azimuth_specs(
                    datasets,
                    frequencies=frequencies,
                    elevation=elevation,
                    polarization=polarization,
                    kind=kind,
                    quantity="magnitude",
                    angle_display_unit="deg",
                    frequency_display_unit="GHz",
                    y_limits=fixed_limits,
                    show_legend=show_legend,
                )
                plots = _with_shared_y_limits(plots, fixed_limits)
                return plan_azimuth_slides(
                    plots,
                    slide_titles=deck_title or "RCS Azimuth Sweeps",
                    footer=footer,
                )
            if kind == "frequency":
                azimuth = self.azimuth_combo.currentData()
                if azimuth is None:
                    raise ValueError("The selected datasets have no common azimuth cut.")
                plot = build_frequency_spec(
                    datasets,
                    azimuth=azimuth,
                    elevation=elevation,
                    polarization=polarization,
                    quantity="magnitude",
                    angle_display_unit="deg",
                    frequency_display_unit="GHz",
                    y_limits=fixed_limits,
                    show_legend=show_legend,
                )
                plot = _with_shared_y_limits((plot,), fixed_limits)[0]
                slide_title = (
                    f"{deck_title} — {plot.title}" if deck_title else plot.title
                )
                return plan_frequency_slides(
                    (plot,), slide_titles=slide_title, footer=footer
                )
            raise ValueError(f"Unsupported PPT plot type: {kind!r}")

        # ------------------------------------------------------------------
        # Preview and export
        # ------------------------------------------------------------------
        def _clear_preview_assets(self) -> None:
            for image_path in Path(self._preview_temp.name).glob("preview_*.png"):
                try:
                    image_path.unlink()
                except OSError:
                    pass

        @Slot()
        def build_preview(self) -> bool:
            if self.job_is_running():
                self._show_error("Wait for the current PowerPoint export to finish.")
                return False
            try:
                plan = self._build_plan()
                self._preview_plan = plan
                self._preview_is_current = True
                self._clear_preview_assets()
                self._preview_generation += 1
                self._current_slide_index = 0
                self._render_current_slide()
            except Exception as exc:
                self._preview_plan = None
                self._preview_is_current = False
                self.export_button.setEnabled(False)
                self._show_error(str(exc))
                self.preview_canvas.show_feedback(
                    "Preview could not be built", str(exc)
                )
                self.page_label.setText("Preview error")
                self._update_navigation()
                return False
            self.export_button.setEnabled(True)
            self._last_error = ""
            message = (
                f"Preview ready: {len(plan.slides)} slide(s), "
                f"{plan.plot_count} plot(s). Review each page, choose an output, "
                "then export PPTX."
            )
            self._set_status(message)
            return True

        def _render_current_slide(self) -> None:
            if self._preview_plan is None:
                return
            slide_count = len(self._preview_plan.slides)
            self._current_slide_index = min(
                max(self._current_slide_index, 0), slide_count - 1
            )
            self.preview_canvas.render_slide(
                self._preview_plan.slides[self._current_slide_index],
                slide_index=self._current_slide_index,
                slide_count=slide_count,
                output_directory=Path(self._preview_temp.name),
                generation=self._preview_generation,
            )
            self.page_label.setText(
                f"Slide {self._current_slide_index + 1} of {slide_count}"
            )
            self._update_navigation()

        @Slot()
        def previous_slide(self) -> None:
            if self._preview_plan is None or self._current_slide_index <= 0:
                return
            self._current_slide_index -= 1
            try:
                self._render_current_slide()
            except Exception as exc:
                self._show_error(str(exc))

        @Slot()
        def next_slide(self) -> None:
            if self._preview_plan is None:
                return
            if self._current_slide_index >= len(self._preview_plan.slides) - 1:
                return
            self._current_slide_index += 1
            try:
                self._render_current_slide()
            except Exception as exc:
                self._show_error(str(exc))

        def _update_navigation(self) -> None:
            usable = self._preview_plan is not None and self._preview_is_current
            count = len(self._preview_plan.slides) if self._preview_plan else 0
            self.previous_slide_button.setEnabled(
                bool(usable and self._current_slide_index > 0)
            )
            self.next_slide_button.setEnabled(
                bool(usable and self._current_slide_index + 1 < count)
            )

        @Slot()
        def export_report(self) -> bool:
            if self.job_is_running():
                self._show_error("A PowerPoint report export is already running.")
                return False
            if self._preview_plan is None or not self._preview_is_current:
                self._show_error("Build and review a current slide preview before export.")
                return False
            output = self.output_edit.text().strip()
            if not output:
                self._show_error("Choose an output .pptx file.")
                return False
            if Path(output).suffix.lower() != ".pptx":
                self._show_error("PowerPoint report output must use the .pptx extension.")
                return False
            output_path = Path(output).expanduser()
            if output_path.exists() and not output_path.is_file():
                self._show_error(f"PowerPoint output is not a file: {output}")
                return False
            if output_path.is_file():
                answer = QMessageBox.question(
                    self,
                    "Replace existing PowerPoint report?",
                    f"The file already exists:\n{output_path}\n\nReplace it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    self._set_status("PowerPoint export canceled; the existing file was kept.")
                    return False
            template = self.template_edit.text().strip()
            if template:
                template_path = Path(template).expanduser()
                if template_path.suffix.lower() not in {".pptx", ".potx"}:
                    self._show_error("Blank templates must use .pptx or .potx.")
                    return False
                if not template_path.is_file():
                    self._show_error(f"Blank PowerPoint template not found: {template}")
                    return False
            thread = QThread(self)
            # PresentationPlan and strings are immutable snapshots. Subsequent
            # shell catalog updates cannot change a report already exporting.
            worker = _ExportWorker(
                self._exporter, self._preview_plan, output, template
            )
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.succeeded.connect(self._export_succeeded)
            worker.failed.connect(self._export_failed)
            worker.succeeded.connect(thread.quit)
            worker.failed.connect(thread.quit)
            worker.succeeded.connect(worker.deleteLater)
            worker.failed.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(self._export_thread_finished)
            self._thread = thread
            self._worker = worker
            self._set_busy(True)
            self._set_status(
                "Rendering fixed-layout plot images and writing the PowerPoint report…"
            )
            thread.start()
            return True

        @Slot(str)
        def _export_succeeded(self, path: str) -> None:
            self._last_error = ""
            self._set_status(f"PowerPoint report saved: {path}")
            self.report_exported.emit(path)

        @Slot(str)
        def _export_failed(self, message: str) -> None:
            self._show_error(message)

        @Slot()
        def _export_thread_finished(self) -> None:
            self._thread = None
            self._worker = None
            self._set_busy(False)

        def _set_busy(self, busy: bool) -> None:
            self.controls_content.setEnabled(not busy)
            self.previous_slide_button.setEnabled(
                not busy
                and self._preview_is_current
                and self._current_slide_index > 0
            )
            count = len(self._preview_plan.slides) if self._preview_plan else 0
            self.next_slide_button.setEnabled(
                not busy
                and self._preview_is_current
                and self._current_slide_index + 1 < count
            )
            if not busy:
                self.export_button.setEnabled(self._preview_is_current)

        def _set_status(self, message: str) -> None:
            text = str(message).strip()
            self.status_label.setText(text)
            self.status_changed.emit(text)

        def _show_error(self, message: str) -> None:
            text = str(message).strip() or "PowerPoint report operation failed."
            self._last_error = text
            self._set_status(text)

        def _browse_template(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Choose blank PowerPoint template",
                self.template_edit.text().strip(),
                "PowerPoint template (*.pptx *.potx);;All files (*)",
            )
            if path:
                self.template_edit.setText(path)

        def _browse_output(self) -> None:
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save PowerPoint report",
                self.output_edit.text().strip() or "RCS_Report.pptx",
                "PowerPoint presentation (*.pptx);;All files (*)",
            )
            if not path:
                return
            if Path(path).suffix == "":
                path += ".pptx"
            self.output_edit.setText(path)

        def closeEvent(self, event: Any) -> None:
            if self.job_is_running():
                self._set_status(
                    "PowerPoint export is still running; wait for it to finish before closing."
                )
                event.ignore()
                return
            self.dispose()
            super().closeEvent(event)

        def dispose(self) -> None:
            """Release preview files when an embedded or standalone host closes."""

            if self._disposed:
                return
            if self._preview_finalizer.alive:
                self._preview_finalizer()
            self._disposed = True


else:

    class PptWorkspace:  # pragma: no cover - exercised only without Qt
        """Placeholder preserving an actionable import-time API."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "PptWorkspace requires PySide6 and Matplotlib Qt support. "
                f"Original import error: {_GUI_IMPORT_ERROR}"
            )


def _plain_scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (ValueError, TypeError):
            pass
    return value


def _axis_values_equal(left: Any, right: Any, tolerance: float = 1.0e-8) -> bool:
    if right is None:
        return False
    try:
        left_float = float(left)
        right_float = float(right)
    except (TypeError, ValueError):
        return str(left) == str(right)
    scale = max(1.0, abs(left_float), abs(right_float))
    return abs(left_float - right_float) <= tolerance * scale


def _find_combo_value(combo: Any, target: Any) -> int:
    if target is None:
        return -1
    for index in range(combo.count()):
        if _axis_values_equal(combo.itemData(index), target):
            return index
    return -1


def _format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.8g}"


def _format_axis_value(value: Any, unit: str) -> str:
    if isinstance(value, str):
        return value
    suffix = str(unit).strip()
    return _format_number(value) + (f" {suffix}" if suffix else "")


def _format_frequency_value(value: Any, unit: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _format_axis_value(value, unit)
    normalized = str(unit).strip().lower()
    hz_scale = {
        "hz": 1.0,
        "khz": 1.0e3,
        "mhz": 1.0e6,
        "ghz": 1.0e9,
    }.get(normalized)
    if hz_scale is None:
        return _format_axis_value(value, unit)
    hz = number * hz_scale
    if abs(hz) >= 1.0e8:
        return f"{hz / 1.0e9:.8g} GHz"
    if abs(hz) >= 1.0e5:
        return f"{hz / 1.0e6:.8g} MHz"
    if abs(hz) >= 1.0e2:
        return f"{hz / 1.0e3:.8g} kHz"
    return f"{hz:.8g} Hz"


__all__ = [
    "DatasetCatalogEntry",
    "GUI_AVAILABLE",
    "PptWorkspace",
]
