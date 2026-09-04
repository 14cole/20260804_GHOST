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
    DUAL_COPOLARIZATION,
    NamedGrid,
    build_azimuth_specs,
    build_elevation_specs,
    build_frequency_specs,
    get_plot_availability,
)
from ppt_report import (
    DEFAULT_AZIMUTH_TEMPLATE_LAYOUT,
    DEFAULT_FREQUENCY_TEMPLATE_LAYOUT,
    PlotSpec,
    PresentationPlan,
    SLIDE_TITLE_FONT_SIZE_POINTS,
    SlidePlan,
    export_powerpoint_report,
    geometry_for_layout,
    plan_azimuth_slides,
    plan_frequency_slides,
    render_master_legend_png,
    render_plot_png,
)


_INITIAL_AZIMUTH_FREQUENCY_COUNT = 6
_MAX_AZIMUTH_REPORT_FREQUENCIES = 60
DEFAULT_POWERPOINT_TEMPLATE = (
    Path(__file__).resolve().parent / "templates" / "GRIM_Report_Template.pptx"
)


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


def _nice_tick_step(low: float, high: float, *, angular: bool = False) -> float:
    """Choose a stable major-tick interval targeting roughly eight divisions."""

    span = float(high) - float(low)
    if not math.isfinite(span) or span <= 0.0:
        return 1.0
    target = span / 8.0
    if angular:
        for candidate in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 45.0, 60.0, 90.0):
            if candidate >= target - 1.0e-12:
                return candidate
    exponent = math.floor(math.log10(target))
    scale = 10.0**exponent
    fraction = target / scale
    if fraction <= 1.0:
        nice_fraction = 1.0
    elif fraction <= 2.0:
        nice_fraction = 2.0
    elif fraction <= 5.0:
        nice_fraction = 5.0
    else:
        nice_fraction = 10.0
    return float(nice_fraction * scale)


def _finite_bounds(values: Iterable[Any]) -> tuple[float, float] | None:
    finite = tuple(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    low, high = min(finite), max(finite)
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1.0e-12):
        padding = max(1.0, abs(low) * 0.05)
        return low - padding, high + padding
    return low, high


_GUI_IMPORT_ERROR: Exception | None = None
try:  # Keep report planning importable on headless/minimal installations.
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.image import imread
    from PySide6.QtCore import QItemSelectionModel, QObject, QThread, Qt, Signal, Slot
    from PySide6.QtGui import QStandardItemModel
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
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
            template_layouts: Mapping[str, str],
        ) -> None:
            super().__init__()
            self._exporter = exporter
            self._plan = plan
            self._output_path = output_path
            self._template_path = template_path
            self._template_layouts = dict(template_layouts)

        @Slot()
        def run(self) -> None:
            try:
                kwargs: dict[str, Any] = {
                    "template_path": self._template_path or None,
                }
                if self._template_layouts:
                    kwargs["template_layouts"] = self._template_layouts
                result = self._exporter(
                    self._plan,
                    self._output_path,
                    **kwargs,
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
            # The master legend intentionally overlaps the top of the plot
            # frames. Create it last and give it an explicit higher z-order so
            # the preview matches PowerPoint's front-most legend layer.
            if slide_plan.master_legend:
                legend_path = output_directory / (
                    f"preview_{generation:04d}_slide_{slide_index + 1:03d}_"
                    "master_legend.png"
                )
                if not legend_path.is_file():
                    render_master_legend_png(
                        slide_plan.master_legend,
                        legend_path,
                        width_points=geometry.master_legend.width,
                        height_points=geometry.master_legend.height,
                        dpi=120,
                    )
                legend = geometry.master_legend
                legend_axes = self.figure.add_axes(
                    (
                        x_position(legend.left),
                        y_position_from_top(legend.bottom),
                        page_width * legend.width / geometry.width,
                        page_height * legend.height / geometry.height,
                    ),
                    label="GRIM master legend",
                )
                legend_axes.set_zorder(100.0)
                legend_axes.set_facecolor("none")
                legend_axes.patch.set_alpha(0.0)
                legend_axes.imshow(imread(legend_path), aspect="auto")
                legend_axes.set_axis_off()
            self.draw_idle()


    class _DatasetOrderListWidget(QListWidget):
        """Internal-move list that never treats another row as a drop target.

        ``QListWidget`` exposes an ``OnItem`` drop mode whenever its rows carry
        ``ItemIsDropEnabled``. Depending on the platform/style, an internal move
        dropped in that zone can be delegated to the item model as a child drop.
        Flat list rows cannot represent that relationship, which made the
        dragged dataset appear to be absorbed by the target row.

        This widget resolves every internal drop to an insertion boundary and
        performs the move itself. Dataset IDs, check states, current item, and
        multi-row selection remain attached to their original item objects.
        """

        order_changed = Signal()

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
            self.setDefaultDropAction(Qt.DropAction.MoveAction)
            self.setDragEnabled(True)
            self.setAcceptDrops(True)
            self.viewport().setAcceptDrops(True)
            self.setDropIndicatorShown(True)

        def _drop_insertion_row(self, event: Any) -> int:
            point = event.position().toPoint()
            target_row = self.indexAt(point).row()
            indicator = self.dropIndicatorPosition()
            if (
                indicator == QAbstractItemView.DropIndicatorPosition.OnViewport
                or target_row < 0
            ):
                return self.count()
            if indicator == QAbstractItemView.DropIndicatorPosition.BelowItem:
                return target_row + 1
            if indicator == QAbstractItemView.DropIndicatorPosition.AboveItem:
                return target_row

            # Defensive handling for styles that still report ``OnItem``:
            # split the row into before/after insertion zones instead of
            # passing an item-parent drop to QListWidget's model.
            target = self.item(target_row)
            rect = self.visualItemRect(target)
            return target_row + int(point.y() >= rect.center().y())

        def move_rows_to_insertion(
            self, source_rows: Iterable[int], insertion_row: int
        ) -> bool:
            """Move flat rows to a pre-removal insertion boundary.

            The helper is intentionally deterministic and independently
            testable because drag/drop event synthesis varies across Qt
            platform plugins. ``insertion_row == count()`` means the viewport
            space after the final row.
            """

            rows = tuple(
                sorted(
                    {
                        int(row)
                        for row in source_rows
                        if 0 <= int(row) < self.count()
                    }
                )
            )
            if not rows:
                return False
            boundary = max(0, min(int(insertion_row), self.count()))
            items = tuple(self.item(row) for row in rows)
            selected_items = tuple(self.selectedItems())
            current_item = self.currentItem()
            before = tuple(self.item(index) for index in range(self.count()))

            for row in reversed(rows):
                self.takeItem(row)
            destination = boundary - sum(row < boundary for row in rows)
            for offset, item in enumerate(items):
                self.insertItem(destination + offset, item)

            self.clearSelection()
            for item in selected_items:
                item.setSelected(True)
            if current_item is not None:
                self.setCurrentItem(
                    current_item,
                    QItemSelectionModel.SelectionFlag.NoUpdate,
                )
            after = tuple(self.item(index) for index in range(self.count()))
            return after != before

        def dropEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
            if event.source() is not self:
                event.ignore()
                return
            rows = tuple(sorted({self.row(item) for item in self.selectedItems()}))
            destination = self._drop_insertion_row(event)
            changed = self.move_rows_to_insertion(rows, destination)
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
            if changed:
                self.order_changed.emit()


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
            self._azimuth_band_axis_signature: tuple[str, tuple[float, ...]] | None = None
            self._azimuth_band_available = False
            self._preview_plan: PresentationPlan | None = None
            self._preview_is_current = False
            self._preview_generation = 0
            self._current_slide_index = 0
            self._thread: QThread | None = None
            self._worker: _ExportWorker | None = None
            self._last_error = ""
            self._last_plan_warnings: tuple[str, ...] = ()
            self._active_x_axis_family = "azimuth"
            self._x_axis_settings: dict[str, tuple[str, float, float, float]] = {
                "azimuth": ("automatic", -180.0, 180.0, 45.0),
                "elevation": ("automatic", -90.0, 90.0, 15.0),
                "frequency": ("automatic", 1.0, 10.0, 1.0),
            }
            self._x_axis_customized: set[str] = set()
            self._series_line_widths: dict[str, float] = {}
            self._series_line_styles: dict[str, str] = {}
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
                    (
                        item.flags()
                        | Qt.ItemFlag.ItemIsUserCheckable
                        | Qt.ItemFlag.ItemIsDragEnabled
                    )
                    & ~Qt.ItemFlag.ItemIsDropEnabled
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
            self._series_line_widths = {
                dataset_id: value
                for dataset_id, value in self._series_line_widths.items()
                if dataset_id in incoming
            }
            self._series_line_styles = {
                dataset_id: value
                for dataset_id, value in self._series_line_styles.items()
                if dataset_id in incoming
            }
            self._refresh_series_style_datasets()
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
            self.dataset_list = _DatasetOrderListWidget(dataset_group)
            self.dataset_list.setObjectName("pptDatasetCatalog")
            self.dataset_list.setMinimumHeight(120)
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
            self.plot_type_combo.addItem("Elevation / pitch sweep", "elevation")
            self.plot_type_combo.addItem("Frequency sweep", "frequency")
            plot_form.addRow("Plot type", self.plot_type_combo)
            self.elevation_label = QLabel("Elevation cut", plot_group)
            self.elevation_combo = QComboBox(plot_group)
            plot_form.addRow(self.elevation_label, self.elevation_combo)
            self.polarization_combo = QComboBox(plot_group)
            self.polarization_combo.setToolTip(
                "'VV and HH' creates separate VV and HH plots in the same "
                "PowerPoint; unlike polarizations are never combined."
            )
            plot_form.addRow("Polarization", self.polarization_combo)
            self.frequency_azimuth_mode_label = QLabel(
                "Frequency trace", plot_group
            )
            self.frequency_azimuth_mode_combo = QComboBox(plot_group)
            self.frequency_azimuth_mode_combo.addItem(
                "Exact azimuth cut", "exact"
            )
            self.frequency_azimuth_mode_combo.addItem(
                "Percentile across azimuth band", "band"
            )
            self.frequency_azimuth_mode_combo.setToolTip(
                "Use one stored azimuth cut, or calculate each frequency point "
                "as a percentile across the selected stored azimuth samples. "
                "Band mode does not interpolate."
            )
            plot_form.addRow(
                self.frequency_azimuth_mode_label,
                self.frequency_azimuth_mode_combo,
            )
            self.azimuth_label = QLabel("Azimuth cut", plot_group)
            self.azimuth_combo = QComboBox(plot_group)
            plot_form.addRow(self.azimuth_label, self.azimuth_combo)
            self.azimuth_band_label = QLabel("Azimuth band", plot_group)
            self.azimuth_band_widget = QWidget(plot_group)
            azimuth_band_layout = QGridLayout(self.azimuth_band_widget)
            azimuth_band_layout.setContentsMargins(0, 0, 0, 0)
            azimuth_band_layout.setHorizontalSpacing(6)
            azimuth_band_layout.setVerticalSpacing(4)
            self.azimuth_band_min_spin = self._axis_spin_box(
                self.azimuth_band_widget, value=-180.0
            )
            self.azimuth_band_max_spin = self._axis_spin_box(
                self.azimuth_band_widget, value=180.0
            )
            self.azimuth_percentile_spin = QDoubleSpinBox(
                self.azimuth_band_widget
            )
            self.azimuth_percentile_spin.setRange(0.0, 100.0)
            self.azimuth_percentile_spin.setDecimals(1)
            self.azimuth_percentile_spin.setSingleStep(5.0)
            self.azimuth_percentile_spin.setValue(90.0)
            self.azimuth_percentile_spin.setSuffix(" %")
            self.azimuth_percentile_spin.setKeyboardTracking(False)
            self.azimuth_percentile_spin.setMinimumWidth(82)
            self.azimuth_band_unit_label = QLabel("deg", self.azimuth_band_widget)
            self.azimuth_band_min_label = QLabel(
                "Minimum", self.azimuth_band_widget
            )
            self.azimuth_band_max_label = QLabel(
                "Maximum", self.azimuth_band_widget
            )
            self.azimuth_percentile_label = QLabel(
                "Percentile", self.azimuth_band_widget
            )
            self.azimuth_band_min_label.setBuddy(self.azimuth_band_min_spin)
            self.azimuth_band_max_label.setBuddy(self.azimuth_band_max_spin)
            self.azimuth_percentile_label.setBuddy(self.azimuth_percentile_spin)
            self.azimuth_band_min_spin.setAccessibleName("Azimuth band minimum")
            self.azimuth_band_max_spin.setAccessibleName("Azimuth band maximum")
            self.azimuth_percentile_spin.setAccessibleName(
                "Azimuth band percentile"
            )
            self.azimuth_band_min_spin.setAccessibleDescription(
                "Inclusive lower endpoint in the displayed dataset angular unit."
            )
            self.azimuth_band_max_spin.setAccessibleDescription(
                "Inclusive upper endpoint. A value below the minimum endpoint "
                "selects across the azimuth seam."
            )
            self.azimuth_percentile_spin.setAccessibleDescription(
                "Percentile calculated across finite stored azimuth samples at "
                "each frequency."
            )
            # Stack the three values instead of forcing both endpoints into
            # one row.  The controls sidebar is intentionally narrow; this
            # remains readable without a horizontal scrollbar at 405 px.
            azimuth_band_layout.addWidget(self.azimuth_band_min_label, 0, 0)
            azimuth_band_layout.addWidget(self.azimuth_band_min_spin, 0, 1)
            azimuth_band_layout.addWidget(self.azimuth_band_unit_label, 0, 2)
            azimuth_band_layout.addWidget(self.azimuth_band_max_label, 1, 0)
            azimuth_band_layout.addWidget(self.azimuth_band_max_spin, 1, 1)
            azimuth_band_layout.addWidget(self.azimuth_percentile_label, 2, 0)
            azimuth_band_layout.addWidget(self.azimuth_percentile_spin, 2, 1)
            azimuth_band_layout.setColumnStretch(1, 1)
            self.azimuth_band_widget.setToolTip(
                "Endpoints are inclusive and use the dataset's stored angular "
                "unit. Min greater than Max selects a wrapped band across the "
                "azimuth seam. The percentile is sample-weighted across common "
                "stored angles: every finite stored angle contributes one sample, "
                "with no angular interpolation. It is calculated independently "
                "at each frequency in displayed dB units."
            )
            plot_form.addRow(self.azimuth_band_label, self.azimuth_band_widget)
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
                "Select up to the first 60 common frequencies (10 angular slides). "
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
            scale_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            self.x_scale_mode_combo = QComboBox(plot_group)
            self.x_scale_mode_combo.addItem("Automatic", "automatic")
            self.x_scale_mode_combo.addItem("Fixed and uniform", "fixed")
            self.x_scale_mode_combo.setToolTip(
                "Fixed values are applied to every plot in the report. Step controls "
                "major tick spacing only; it does not resample dataset values."
            )
            self.x_scale_label = QLabel("Horizontal axis (deg)", plot_group)
            scale_form.addRow(self.x_scale_label, self.x_scale_mode_combo)
            self.fixed_x_scale_widget = QWidget(plot_group)
            fixed_x_layout = QGridLayout(self.fixed_x_scale_widget)
            fixed_x_layout.setContentsMargins(0, 0, 0, 0)
            fixed_x_layout.setHorizontalSpacing(6)
            fixed_x_layout.setVerticalSpacing(3)
            self.x_min_spin = self._axis_spin_box(
                self.fixed_x_scale_widget, value=-180.0
            )
            self.x_max_spin = self._axis_spin_box(
                self.fixed_x_scale_widget, value=180.0
            )
            self.x_step_spin = self._axis_spin_box(
                self.fixed_x_scale_widget, value=45.0, positive=True
            )
            for column, (label, spin) in enumerate((
                ("Min", self.x_min_spin),
                ("Max", self.x_max_spin),
                ("Step", self.x_step_spin),
            )):
                label_widget = QLabel(label, self.fixed_x_scale_widget)
                label_widget.setBuddy(spin)
                fixed_x_layout.addWidget(label_widget, 0, column)
                fixed_x_layout.addWidget(spin, 1, column)
            self.x_data_bounds_button = QPushButton(
                "Use data bounds", self.fixed_x_scale_widget
            )
            self.x_data_bounds_button.setToolTip(
                "Set the minimum, maximum, and a readable major-tick step from "
                "the selected datasets. No data are resampled."
            )
            fixed_x_layout.addWidget(self.x_data_bounds_button, 2, 0, 1, 3)
            scale_form.addRow("Fixed horizontal", self.fixed_x_scale_widget)

            self.scale_mode_combo = QComboBox(plot_group)
            self.scale_mode_combo.addItem("Shared automatic", "shared_auto")
            self.scale_mode_combo.addItem("Fixed and uniform", "fixed")
            self.scale_mode_combo.setToolTip(
                "Every plot uses one vertical RCS scale. Fixed mode also anchors "
                "major ticks at the supplied minimum using the selected step."
            )
            scale_form.addRow("Vertical RCS axis", self.scale_mode_combo)
            self.fixed_scale_widget = QWidget(plot_group)
            fixed_scale_layout = QGridLayout(self.fixed_scale_widget)
            fixed_scale_layout.setContentsMargins(0, 0, 0, 0)
            fixed_scale_layout.setHorizontalSpacing(6)
            fixed_scale_layout.setVerticalSpacing(3)
            self.y_min_spin = self._axis_spin_box(
                self.fixed_scale_widget, value=-60.0
            )
            self.y_max_spin = self._axis_spin_box(
                self.fixed_scale_widget, value=20.0
            )
            self.y_step_spin = self._axis_spin_box(
                self.fixed_scale_widget, value=10.0, positive=True
            )
            for column, (label, spin) in enumerate((
                ("Min", self.y_min_spin),
                ("Max", self.y_max_spin),
                ("Step", self.y_step_spin),
            )):
                label_widget = QLabel(label, self.fixed_scale_widget)
                label_widget.setBuddy(spin)
                fixed_scale_layout.addWidget(label_widget, 0, column)
                fixed_scale_layout.addWidget(spin, 1, column)
            self.y_data_bounds_button = QPushButton(
                "Use data bounds", self.fixed_scale_widget
            )
            self.y_data_bounds_button.setToolTip(
                "Build the selected cuts in memory and choose a shared readable "
                "vertical range from every finite plotted value."
            )
            fixed_scale_layout.addWidget(self.y_data_bounds_button, 2, 0, 1, 3)
            scale_form.addRow("Fixed vertical", self.fixed_scale_widget)
            self.legend_mode_combo = QComboBox(plot_group)
            self.legend_mode_combo.addItem("Master across slide", "master")
            self.legend_mode_combo.addItem("Inside each plot", "per_plot")
            self.legend_mode_combo.addItem("None", "none")
            self.legend_mode_combo.setToolTip(
                "The master legend appears once beneath the slide title and follows "
                "the dataset order above."
            )
            scale_form.addRow("Dataset legend", self.legend_mode_combo)
            self.global_line_width_spin = QDoubleSpinBox(plot_group)
            self.global_line_width_spin.setRange(0.5, 5.0)
            self.global_line_width_spin.setDecimals(2)
            self.global_line_width_spin.setSingleStep(0.25)
            self.global_line_width_spin.setValue(1.5)
            self.global_line_width_spin.setSuffix(" pt")
            self.global_line_width_spin.setKeyboardTracking(False)
            self.global_line_width_spin.setToolTip(
                "Default plotted line thickness for every dataset. Individual "
                "datasets can override it below."
            )
            scale_form.addRow("Line thickness", self.global_line_width_spin)
            self.global_line_style_combo = QComboBox(plot_group)
            self.global_line_style_combo.addItem("Solid", "-")
            self.global_line_style_combo.addItem("Dashed", "--")
            self.global_line_style_combo.addItem("Dotted", ":")
            self.global_line_style_combo.addItem("Dash-dot", "-.")
            scale_form.addRow("Line style", self.global_line_style_combo)
            plot_layout.addLayout(scale_form)

            self.series_override_button = QPushButton(
                "Series overrides…", plot_group
            )
            self.series_override_button.setCheckable(True)
            self.series_override_button.setChecked(False)
            self.series_override_button.setToolTip(
                "Optionally override thickness or pattern for one loaded dataset."
            )
            plot_layout.addWidget(self.series_override_button)
            self.series_override_widget = QWidget(plot_group)
            series_override_form = QFormLayout(self.series_override_widget)
            series_override_form.setContentsMargins(0, 0, 0, 0)
            self.series_dataset_combo = QComboBox(self.series_override_widget)
            series_override_form.addRow("Dataset", self.series_dataset_combo)
            self.series_line_width_spin = QDoubleSpinBox(self.series_override_widget)
            self.series_line_width_spin.setRange(0.0, 5.0)
            self.series_line_width_spin.setDecimals(2)
            self.series_line_width_spin.setSingleStep(0.25)
            self.series_line_width_spin.setSpecialValueText("Use global")
            self.series_line_width_spin.setSuffix(" pt")
            self.series_line_width_spin.setKeyboardTracking(False)
            series_override_form.addRow("Thickness", self.series_line_width_spin)
            self.series_line_style_combo = QComboBox(self.series_override_widget)
            self.series_line_style_combo.addItem("Use global", "")
            self.series_line_style_combo.addItem("Solid", "-")
            self.series_line_style_combo.addItem("Dashed", "--")
            self.series_line_style_combo.addItem("Dotted", ":")
            self.series_line_style_combo.addItem("Dash-dot", "-.")
            series_override_form.addRow("Pattern", self.series_line_style_combo)
            self.reset_series_style_button = QPushButton(
                "Reset selected dataset", self.series_override_widget
            )
            series_override_form.addRow("", self.reset_series_style_button)
            self.series_override_widget.setVisible(False)
            plot_layout.addWidget(self.series_override_widget)

            self.axis_warning_label = QLabel("", plot_group)
            self.axis_warning_label.setObjectName("pptAxisWarning")
            self.axis_warning_label.setWordWrap(True)
            self.axis_warning_label.setVisible(False)
            plot_layout.addWidget(self.axis_warning_label)
            controls.addWidget(plot_group)

            deck_group = QGroupBox("3  Slide text and files", self.controls_content)
            deck_layout = QVBoxLayout(deck_group)
            deck_form = QFormLayout()
            deck_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            self.deck_title_edit = QLineEdit("RCS Report", deck_group)
            self.deck_title_edit.setPlaceholderText("RCS Report")
            deck_form.addRow("Slide title", self.deck_title_edit)
            self.output_edit, output_widget = self._path_row(
                deck_group, "Choose report output", self._browse_output
            )
            self.output_edit.setText("RCS_Report.pptx")
            deck_form.addRow("Output .pptx", output_widget)
            deck_layout.addLayout(deck_form)

            self.template_options_button = QPushButton(
                "Template and layouts…", deck_group
            )
            self.template_options_button.setCheckable(True)
            self.template_options_button.setChecked(False)
            deck_layout.addWidget(self.template_options_button)
            self.template_options_widget = QWidget(deck_group)
            template_form = QFormLayout(self.template_options_widget)
            template_form.setContentsMargins(0, 0, 0, 0)
            template_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            self.template_edit, template_widget = self._path_row(
                self.template_options_widget,
                "Choose PowerPoint template",
                self._browse_template,
            )
            self.template_edit.setPlaceholderText("Optional .pptx or .potx template")
            self.template_edit.setToolTip(
                "Optional widescreen 16:9 .pptx or .potx. GRIM clears example "
                "slides but retains themes, masters, graphics, and named layouts."
            )
            if DEFAULT_POWERPOINT_TEMPLATE.is_file():
                self.template_edit.setText(str(DEFAULT_POWERPOINT_TEMPLATE))
            template_form.addRow("PowerPoint template", template_widget)
            self.azimuth_layout_edit = QLineEdit(
                DEFAULT_AZIMUTH_TEMPLATE_LAYOUT,
                self.template_options_widget,
            )
            self.azimuth_layout_edit.setPlaceholderText(
                "Leave blank for PowerPoint's generic blank layout"
            )
            self.azimuth_layout_edit.setToolTip(
                "Named custom layout for azimuth, polar, and elevation reports. "
                "Use 'Master :: Layout' when the same layout name appears under "
                "more than one slide master."
            )
            template_form.addRow("Angular custom layout", self.azimuth_layout_edit)
            self.frequency_layout_edit = QLineEdit(
                DEFAULT_FREQUENCY_TEMPLATE_LAYOUT,
                self.template_options_widget,
            )
            self.frequency_layout_edit.setPlaceholderText(
                "Leave blank for PowerPoint's generic blank layout"
            )
            self.frequency_layout_edit.setToolTip(
                "Named custom layout for one-plot frequency-sweep slides. Use "
                "'Master :: Layout' to qualify duplicate layout names."
            )
            template_form.addRow("Frequency custom layout", self.frequency_layout_edit)
            self.template_options_widget.setVisible(False)
            deck_layout.addWidget(self.template_options_widget)
            controls.addWidget(deck_group)

            controls.addStretch(1)
            self.controls_scroll.setWidget(self.controls_content)
            splitter.addWidget(self.controls_scroll)

            preview_panel = QWidget(splitter)
            preview_layout = QVBoxLayout(preview_panel)
            preview_layout.setContentsMargins(8, 0, 0, 0)
            preview_header = QHBoxLayout()
            preview_title = QLabel("Plot-placement preview", preview_panel)
            preview_title.setStyleSheet("font-weight: 600;")
            preview_title.setToolTip(
                "Shows the exact GRIM plot, title, and legend rectangles. "
                "Graphics inherited from a selected PowerPoint master appear "
                "in the exported PPTX, not on this lightweight preview canvas."
            )
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

            action_bar = QHBoxLayout()
            action_bar.setContentsMargins(0, 2, 0, 0)
            action_bar.addStretch(1)
            self.build_preview_button = QPushButton("Build preview", self)
            self.build_preview_button.setDefault(True)
            self.export_button = QPushButton("Export PowerPoint", self)
            self.export_button.setEnabled(False)
            action_bar.addWidget(self.build_preview_button)
            action_bar.addWidget(self.export_button)
            outer.addLayout(action_bar)
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

        @staticmethod
        def _axis_spin_box(
            parent: QWidget,
            *,
            value: float,
            positive: bool = False,
        ) -> QDoubleSpinBox:
            spin = QDoubleSpinBox(parent)
            spin.setRange(1.0e-6 if positive else -1.0e9, 1.0e9)
            spin.setDecimals(6)
            spin.setValue(float(value))
            spin.setSingleStep(1.0)
            spin.setKeyboardTracking(False)
            spin.setMinimumWidth(82)
            spin.setMaximumWidth(124)
            return spin

        def _connect_signals(self) -> None:
            self.dataset_list.itemChanged.connect(self._dataset_selection_changed)
            self.dataset_list.order_changed.connect(self._dataset_selection_changed)
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
                self.frequency_azimuth_mode_combo,
                self.x_scale_mode_combo,
                self.scale_mode_combo,
                self.legend_mode_combo,
            ):
                combo.currentIndexChanged.connect(self._control_changed)
            for spin in (
                self.x_min_spin,
                self.x_max_spin,
                self.x_step_spin,
                self.y_min_spin,
                self.y_max_spin,
                self.y_step_spin,
                self.azimuth_band_min_spin,
                self.azimuth_band_max_spin,
                self.azimuth_percentile_spin,
            ):
                spin.valueChanged.connect(self._mark_preview_stale)
            for spin in (self.x_min_spin, self.x_max_spin, self.x_step_spin):
                spin.valueChanged.connect(self._x_axis_value_changed)
            self.x_data_bounds_button.clicked.connect(self._use_data_x_bounds)
            self.y_data_bounds_button.clicked.connect(self._use_data_y_bounds)
            self.global_line_width_spin.valueChanged.connect(self._mark_preview_stale)
            self.global_line_style_combo.currentIndexChanged.connect(
                self._mark_preview_stale
            )
            self.series_override_button.toggled.connect(
                self.series_override_widget.setVisible
            )
            self.series_dataset_combo.currentIndexChanged.connect(
                self._load_series_style_controls
            )
            self.series_line_width_spin.valueChanged.connect(
                self._store_series_style_override
            )
            self.series_line_style_combo.currentIndexChanged.connect(
                self._store_series_style_override
            )
            self.reset_series_style_button.clicked.connect(
                self._reset_series_style_override
            )
            self.deck_title_edit.textChanged.connect(self._mark_preview_stale)
            self.template_edit.textChanged.connect(self._mark_preview_stale)
            self.template_edit.textChanged.connect(self._update_template_controls)
            self.azimuth_layout_edit.textChanged.connect(self._mark_preview_stale)
            self.frequency_layout_edit.textChanged.connect(self._mark_preview_stale)
            self.template_options_button.toggled.connect(
                self.template_options_widget.setVisible
            )
            self.build_preview_button.clicked.connect(self.build_preview)
            self.export_button.clicked.connect(self.export_report)
            self.previous_slide_button.clicked.connect(self.previous_slide)
            self.next_slide_button.clicked.connect(self.next_slide)
            self._update_template_controls()

        def _set_azimuth_band_available(
            self, available: bool, *, reason: str = ""
        ) -> None:
            """Enable band mode only when a percentile has multiple samples."""

            self._azimuth_band_available = bool(available)
            band_index = self.frequency_azimuth_mode_combo.findData("band")
            model = self.frequency_azimuth_mode_combo.model()
            if band_index >= 0 and isinstance(model, QStandardItemModel):
                item = model.item(band_index)
                if item is not None:
                    item.setEnabled(self._azimuth_band_available)
                    item.setToolTip(
                        "Calculate a percentile across a common azimuth band."
                        if self._azimuth_band_available
                        else reason
                    )

            base_tooltip = (
                "Use one stored azimuth cut, or calculate each frequency point "
                "as a sample-weighted percentile across common stored azimuth "
                "angles. Every stored angle contributes one sample; band mode "
                "does not interpolate angles."
            )
            self.frequency_azimuth_mode_combo.setToolTip(
                base_tooltip
                if self._azimuth_band_available or not reason
                else f"{base_tooltip}\n\nBand unavailable: {reason}"
            )
            if (
                not self._azimuth_band_available
                and self.frequency_azimuth_mode_combo.currentData() == "band"
            ):
                exact_index = self.frequency_azimuth_mode_combo.findData("exact")
                self.frequency_azimuth_mode_combo.blockSignals(True)
                self.frequency_azimuth_mode_combo.setCurrentIndex(exact_index)
                self.frequency_azimuth_mode_combo.blockSignals(False)
                self._control_changed()

        def _update_template_controls(self, *_args: Any) -> None:
            enabled = bool(self.template_edit.text().strip())
            self.azimuth_layout_edit.setEnabled(enabled)
            self.frequency_layout_edit.setEnabled(enabled)

        def _selected_template_layouts(self) -> dict[str, str]:
            if not self.template_edit.text().strip():
                return {}
            values = {
                "azimuth_3x2": self.azimuth_layout_edit.text().strip(),
                "frequency_single": self.frequency_layout_edit.text().strip(),
            }
            return {kind: selector for kind, selector in values.items() if selector}

        def _refresh_series_style_datasets(self) -> None:
            current = self.series_dataset_combo.currentData()
            self.series_dataset_combo.blockSignals(True)
            self.series_dataset_combo.clear()
            for dataset_id in self.dataset_ids_in_order():
                entry = self._catalog.get(dataset_id)
                if entry is not None:
                    self.series_dataset_combo.addItem(entry.name, dataset_id)
            index = self.series_dataset_combo.findData(current)
            if index < 0 and self.series_dataset_combo.count():
                index = 0
            self.series_dataset_combo.setCurrentIndex(index)
            self.series_dataset_combo.blockSignals(False)
            self._load_series_style_controls()

        @Slot()
        def _load_series_style_controls(self, *_args: Any) -> None:
            dataset_id = str(self.series_dataset_combo.currentData() or "")
            self.series_line_width_spin.blockSignals(True)
            self.series_line_style_combo.blockSignals(True)
            try:
                self.series_line_width_spin.setValue(
                    float(self._series_line_widths.get(dataset_id, 0.0))
                )
                style = self._series_line_styles.get(dataset_id, "")
                index = self.series_line_style_combo.findData(style)
                self.series_line_style_combo.setCurrentIndex(max(index, 0))
            finally:
                self.series_line_width_spin.blockSignals(False)
                self.series_line_style_combo.blockSignals(False)
            enabled = bool(dataset_id)
            self.series_line_width_spin.setEnabled(enabled)
            self.series_line_style_combo.setEnabled(enabled)
            self.reset_series_style_button.setEnabled(enabled)

        @Slot()
        def _store_series_style_override(self, *_args: Any) -> None:
            if self._syncing:
                return
            dataset_id = str(self.series_dataset_combo.currentData() or "")
            if not dataset_id:
                return
            width = float(self.series_line_width_spin.value())
            if width > 0.0:
                self._series_line_widths[dataset_id] = width
            else:
                self._series_line_widths.pop(dataset_id, None)
            style = str(self.series_line_style_combo.currentData() or "")
            if style:
                self._series_line_styles[dataset_id] = style
            else:
                self._series_line_styles.pop(dataset_id, None)
            self._mark_preview_stale()

        @Slot()
        def _reset_series_style_override(self) -> None:
            dataset_id = str(self.series_dataset_combo.currentData() or "")
            if not dataset_id:
                return
            self._series_line_widths.pop(dataset_id, None)
            self._series_line_styles.pop(dataset_id, None)
            self._load_series_style_controls()
            self._mark_preview_stale()

        def _apply_series_styles(
            self,
            plots: Sequence[PlotSpec],
            entries: Sequence[DatasetCatalogEntry],
        ) -> tuple[PlotSpec, ...]:
            global_width = float(self.global_line_width_spin.value())
            global_style = str(self.global_line_style_combo.currentData() or "-")
            values: list[PlotSpec] = []
            for plot in plots:
                styled_series = []
                for index, series in enumerate(plot.series):
                    dataset_id = (
                        entries[index].dataset_id if index < len(entries) else ""
                    )
                    styled_series.append(
                        replace(
                            series,
                            line_width=float(
                                self._series_line_widths.get(
                                    dataset_id,
                                    global_width,
                                )
                            ),
                            line_style=self._series_line_styles.get(
                                dataset_id,
                                global_style,
                            ),
                        )
                    )
                values.append(replace(plot, series=tuple(styled_series)))
            return tuple(values)

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
                self._azimuth_band_axis_signature = None
                self._set_azimuth_band_available(
                    False,
                    reason="Load a dataset with at least two common azimuth samples.",
                )
                self.azimuth_label.setText("Azimuth cut")
                self.azimuth_band_label.setText("Azimuth band")
                self.elevation_label.setText("Elevation cut")
                self._set_axis_combo(self.elevation_combo, (), "")
                self._set_axis_combo(self.azimuth_combo, (), "")
                self._set_polarization_combo(())
                self._set_frequency_choices((), "")
                return
            try:
                availability = get_plot_availability(
                    self._named_grids(),
                    evaluate_phase=False,
                )
            except Exception as exc:
                self._availability = None
                self._azimuth_band_axis_signature = None
                self._set_azimuth_band_available(
                    False,
                    reason="Selected datasets do not have a usable common azimuth axis.",
                )
                self.azimuth_label.setText("Azimuth cut")
                self.azimuth_band_label.setText("Azimuth band")
                self.elevation_label.setText("Elevation cut")
                self.dataset_summary_label.setText(
                    f"Selected datasets are not plot-compatible: {exc}"
                )
                self._set_axis_combo(self.elevation_combo, (), "")
                self._set_axis_combo(self.azimuth_combo, (), "")
                self._set_polarization_combo(())
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
            self.azimuth_band_label.setText(
                "Aspect band" if great_circle else "Azimuth band"
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
            self._set_polarization_combo(
                tuple(availability.polarizations), preferred="HH"
            )
            self._sync_azimuth_band_axis(
                tuple(availability.azimuths), angle_unit
            )
            self._set_frequency_choices(
                tuple(availability.frequencies), frequency_unit
            )
            self._refresh_axis_data_defaults(availability)
            rcs_suffix = f" {availability.rcs_unit}"
            for spin in (self.y_min_spin, self.y_max_spin, self.y_step_spin):
                spin.setSuffix(rcs_suffix)

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

        def _set_polarization_combo(
            self,
            values: Sequence[Any],
            *,
            preferred: Any = None,
        ) -> None:
            current = self.polarization_combo.currentData()
            available = tuple(str(value) for value in values)
            copolar = tuple(
                value for value in available if value in {"VV", "HH"}
            )
            remaining = tuple(
                value for value in available if value not in {"VV", "HH"}
            )
            ordered: tuple[str, ...]
            if set(copolar) == {"VV", "HH"}:
                # Keep the report convenience directly after the two
                # independent co-polar choices.  Cross-polar channels, when
                # present, remain available after it.
                ordered = (*copolar, DUAL_COPOLARIZATION, *remaining)
            else:
                ordered = available

            self.polarization_combo.blockSignals(True)
            self.polarization_combo.clear()
            for value in ordered:
                self.polarization_combo.addItem(value, value)
            target = current if current is not None else preferred
            target_index = self.polarization_combo.findData(target)
            if target_index < 0 and self.polarization_combo.count():
                target_index = 0
            self.polarization_combo.setCurrentIndex(target_index)
            self.polarization_combo.blockSignals(False)

        def _sync_azimuth_band_axis(
            self, values: Sequence[Any], unit: str
        ) -> None:
            numeric = tuple(float(value) for value in values)
            unit_text = str(unit or "deg")
            self.azimuth_band_unit_label.setText(unit_text)
            axis_name = (
                "Aspect"
                if self.azimuth_band_label.text().casefold().startswith("aspect")
                else "Azimuth"
            )
            self.azimuth_band_min_spin.setAccessibleName(
                f"{axis_name} band minimum ({unit_text})"
            )
            self.azimuth_band_max_spin.setAccessibleName(
                f"{axis_name} band maximum ({unit_text})"
            )
            is_radian = unit_text.strip().casefold() in {"rad", "radian", "radians"}
            decimals = 10 if is_radian else 6
            self.azimuth_band_min_spin.setDecimals(decimals)
            self.azimuth_band_max_spin.setDecimals(decimals)
            unique_count = _physical_angle_sample_count(numeric, unit_text)
            self._set_azimuth_band_available(
                unique_count >= 2,
                reason=(
                    "At least two common azimuth samples are required."
                    if numeric
                    else "No common azimuth samples are available."
                ),
            )
            signature = (unit_text.casefold(), numeric)
            if not numeric or signature == self._azimuth_band_axis_signature:
                return

            widgets = (
                self.azimuth_band_min_spin,
                self.azimuth_band_max_spin,
            )
            for widget in widgets:
                widget.blockSignals(True)
            try:
                axis_min = min(numeric)
                axis_max = max(numeric)
                for widget in widgets:
                    widget.setRange(axis_min, axis_max)
                self.azimuth_band_min_spin.setValue(axis_min)
                self.azimuth_band_max_spin.setValue(axis_max)
                ordered = sorted(set(numeric))
                if len(ordered) > 1:
                    positive_steps = [
                        right - left
                        for left, right in zip(ordered, ordered[1:])
                        if right > left
                    ]
                    if positive_steps:
                        step = min(positive_steps)
                        self.azimuth_band_min_spin.setSingleStep(step)
                        self.azimuth_band_max_spin.setSingleStep(step)
            finally:
                for widget in widgets:
                    widget.blockSignals(False)
            self._azimuth_band_axis_signature = signature

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

        @staticmethod
        def _display_axis_values(
            values: Sequence[Any], unit: str, family: str
        ) -> tuple[float, ...]:
            unit_text = str(unit or "").strip().casefold()
            numeric = tuple(float(value) for value in values)
            if family in {"azimuth", "elevation"}:
                if unit_text in {"rad", "radian", "radians"}:
                    return tuple(math.degrees(value) for value in numeric)
                return numeric
            if family == "frequency":
                factors = {
                    "hz": 1.0e-9,
                    "khz": 1.0e-6,
                    "mhz": 1.0e-3,
                    "ghz": 1.0,
                    "thz": 1.0e3,
                }
                factor = factors.get(unit_text)
                if factor is None:
                    raise ValueError(
                        f"Unsupported frequency display conversion from {unit!r}."
                    )
                return tuple(value * factor for value in numeric)
            raise ValueError(f"Unsupported horizontal-axis family: {family!r}")

        def _axis_data_values(self, family: str) -> tuple[float, ...]:
            availability = self._availability
            if availability is None:
                return ()
            if family == "azimuth":
                return self._display_axis_values(
                    tuple(availability.azimuths),
                    str(availability.azimuth_unit),
                    family,
                )
            if family == "elevation":
                return self._display_axis_values(
                    tuple(availability.elevations),
                    str(availability.elevation_unit),
                    family,
                )
            return self._display_axis_values(
                tuple(availability.frequencies),
                str(availability.frequency_unit),
                family,
            )

        def _refresh_axis_data_defaults(self, availability: Any) -> None:
            del availability  # The normalized values are read through self._availability.
            for family in ("azimuth", "elevation", "frequency"):
                if family in self._x_axis_customized:
                    continue
                bounds = _finite_bounds(self._axis_data_values(family))
                if bounds is None:
                    continue
                low, high = bounds
                mode = self._x_axis_settings[family][0]
                self._x_axis_settings[family] = (
                    mode,
                    low,
                    high,
                    _nice_tick_step(
                        low,
                        high,
                        angular=family in {"azimuth", "elevation"},
                    ),
                )
            if self._active_x_axis_family in self._x_axis_customized:
                return
            mode, low, high, step = self._x_axis_settings[
                self._active_x_axis_family
            ]
            widgets = (
                self.x_scale_mode_combo,
                self.x_min_spin,
                self.x_max_spin,
                self.x_step_spin,
            )
            for widget in widgets:
                widget.blockSignals(True)
            try:
                mode_index = self.x_scale_mode_combo.findData(mode)
                self.x_scale_mode_combo.setCurrentIndex(max(mode_index, 0))
                self.x_min_spin.setValue(low)
                self.x_max_spin.setValue(high)
                self.x_step_spin.setValue(step)
            finally:
                for widget in widgets:
                    widget.blockSignals(False)

        @Slot()
        def _x_axis_value_changed(self, *_args: Any) -> None:
            if self._syncing:
                return
            self._x_axis_customized.add(self._active_x_axis_family)
            self._x_axis_settings[self._active_x_axis_family] = (
                str(self.x_scale_mode_combo.currentData() or "automatic"),
                float(self.x_min_spin.value()),
                float(self.x_max_spin.value()),
                float(self.x_step_spin.value()),
            )

        @Slot()
        def _use_data_x_bounds(self) -> None:
            try:
                bounds = _finite_bounds(
                    self._axis_data_values(self._active_x_axis_family)
                )
                if bounds is None:
                    raise ValueError(
                        "The selected datasets have no finite horizontal-axis values."
                    )
                low, high = bounds
                step = _nice_tick_step(
                    low,
                    high,
                    angular=self._active_x_axis_family
                    in {"azimuth", "elevation"},
                )
            except Exception as exc:
                self._show_error(str(exc))
                return
            widgets = (
                self.x_scale_mode_combo,
                self.x_min_spin,
                self.x_max_spin,
                self.x_step_spin,
            )
            for widget in widgets:
                widget.blockSignals(True)
            try:
                self.x_scale_mode_combo.setCurrentIndex(
                    self.x_scale_mode_combo.findData("fixed")
                )
                self.x_min_spin.setValue(low)
                self.x_max_spin.setValue(high)
                self.x_step_spin.setValue(step)
            finally:
                for widget in widgets:
                    widget.blockSignals(False)
            self._x_axis_customized.discard(self._active_x_axis_family)
            self._x_axis_settings[self._active_x_axis_family] = (
                "fixed",
                low,
                high,
                step,
            )
            self._control_changed()

        @Slot()
        def _use_data_y_bounds(self) -> None:
            original_index = self.scale_mode_combo.currentIndex()
            self.scale_mode_combo.blockSignals(True)
            try:
                automatic_index = self.scale_mode_combo.findData("shared_auto")
                self.scale_mode_combo.setCurrentIndex(automatic_index)
                plan = self._build_plan()
                plots = tuple(
                    placement.plot
                    for slide in plan.slides
                    for placement in slide.plots
                )
                low, high = _finite_common_y_limits(plots)
                step = _nice_tick_step(low, high)
            except Exception as exc:
                self.scale_mode_combo.setCurrentIndex(original_index)
                self.scale_mode_combo.blockSignals(False)
                self._show_error(str(exc))
                return
            self.scale_mode_combo.setCurrentIndex(
                self.scale_mode_combo.findData("fixed")
            )
            self.scale_mode_combo.blockSignals(False)
            for spin in (self.y_min_spin, self.y_max_spin, self.y_step_spin):
                spin.blockSignals(True)
            try:
                self.y_min_spin.setValue(low)
                self.y_max_spin.setValue(high)
                self.y_step_spin.setValue(step)
            finally:
                for spin in (self.y_min_spin, self.y_max_spin, self.y_step_spin):
                    spin.blockSignals(False)
            self._control_changed()

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
                    "one report to 60 frequencies (10 angular slides) to keep preview "
                    "and export responsive."
                )

        @Slot()
        def _update_plot_type_controls(self, *_args: Any) -> None:
            kind = str(self.plot_type_combo.currentData() or "azimuth_rect")
            angular_kind = kind in {"azimuth_rect", "azimuth_polar", "elevation"}
            x_family = (
                "azimuth"
                if kind in {"azimuth_rect", "azimuth_polar"}
                else "elevation"
                if kind == "elevation"
                else "frequency"
            )
            if x_family != self._active_x_axis_family:
                self._x_axis_settings[self._active_x_axis_family] = (
                    str(self.x_scale_mode_combo.currentData() or "automatic"),
                    float(self.x_min_spin.value()),
                    float(self.x_max_spin.value()),
                    float(self.x_step_spin.value()),
                )
                mode, low, high, step = self._x_axis_settings[x_family]
                widgets = (
                    self.x_scale_mode_combo,
                    self.x_min_spin,
                    self.x_max_spin,
                    self.x_step_spin,
                )
                for widget in widgets:
                    widget.blockSignals(True)
                try:
                    mode_index = self.x_scale_mode_combo.findData(mode)
                    self.x_scale_mode_combo.setCurrentIndex(max(mode_index, 0))
                    self.x_min_spin.setValue(low)
                    self.x_max_spin.setValue(high)
                    self.x_step_spin.setValue(step)
                finally:
                    for widget in widgets:
                        widget.blockSignals(False)
                self._active_x_axis_family = x_family
            self.frequency_box.setVisible(angular_kind)
            self.frequency_azimuth_mode_label.setVisible(kind == "frequency")
            self.frequency_azimuth_mode_combo.setVisible(kind == "frequency")
            self.x_scale_label.setText(
                "Horizontal axis (deg)" if angular_kind else "Horizontal axis (GHz)"
            )
            x_suffix = " deg" if angular_kind else " GHz"
            for spin in (self.x_min_spin, self.x_max_spin, self.x_step_spin):
                spin.setSuffix(x_suffix)
            self._control_changed()

        @Slot()
        def _control_changed(self, *_args: Any) -> None:
            frequency_kind = (
                str(self.plot_type_combo.currentData() or "") == "frequency"
            )
            band_mode = (
                str(self.frequency_azimuth_mode_combo.currentData() or "exact")
                == "band"
            )
            elevation_kind = (
                str(self.plot_type_combo.currentData() or "") == "elevation"
            )
            self.azimuth_label.setVisible(
                elevation_kind or (frequency_kind and not band_mode)
            )
            self.azimuth_combo.setVisible(
                elevation_kind or (frequency_kind and not band_mode)
            )
            self.azimuth_band_label.setVisible(frequency_kind and band_mode)
            self.azimuth_band_widget.setVisible(frequency_kind and band_mode)
            self.elevation_label.setVisible(not elevation_kind)
            self.elevation_combo.setVisible(not elevation_kind)
            self.fixed_x_scale_widget.setVisible(
                self.x_scale_mode_combo.currentData() == "fixed"
            )
            self.fixed_scale_widget.setVisible(
                self.scale_mode_combo.currentData() == "fixed"
            )
            self._mark_preview_stale()

        @Slot()
        def _mark_preview_stale(self, *_args: Any) -> None:
            self.axis_warning_label.clear()
            self.axis_warning_label.setVisible(False)
            if self._syncing or not self._preview_is_current:
                return
            self._preview_is_current = False
            self.export_button.setEnabled(False)
            message = "Settings changed — click Build Preview to verify the updated slides."
            self.preview_canvas.show_feedback("Preview is out of date", message)
            self.page_label.setText("Stale preview")
            self._set_status(message)
            self._update_navigation()

        @staticmethod
        def _validated_fixed_axis(
            mode: str,
            low: float,
            high: float,
            step: float,
            *,
            axis_name: str,
            maximum_span: float | None = None,
        ) -> tuple[tuple[float, float] | None, float | None]:
            if mode != "fixed":
                return None, None
            values = (float(low), float(high), float(step))
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Fixed {axis_name} values must be finite.")
            low_value, high_value, step_value = values
            if low_value >= high_value:
                raise ValueError(
                    f"Fixed {axis_name} minimum must be less than the maximum."
                )
            if step_value <= 0:
                raise ValueError(f"Fixed {axis_name} step must be positive.")
            span = high_value - low_value
            if maximum_span is not None and span > maximum_span + 1.0e-9:
                raise ValueError(
                    f"Fixed {axis_name} range may span at most {maximum_span:g}."
                )
            tick_count = int(math.floor(span / step_value + 1.0e-9)) + 1
            if tick_count > 1_000:
                raise ValueError(
                    f"Fixed {axis_name} step would create more than 1,000 ticks."
                )
            return (low_value, high_value), step_value

        def _axis_overrides(
            self, kind: str
        ) -> tuple[
            tuple[float, float] | None,
            float | None,
            tuple[float, float] | None,
            float | None,
        ]:
            x_limits, x_step = self._validated_fixed_axis(
                str(self.x_scale_mode_combo.currentData() or "automatic"),
                self.x_min_spin.value(),
                self.x_max_spin.value(),
                self.x_step_spin.value(),
                axis_name=(
                    "azimuth axis (deg)"
                    if kind in {"azimuth_rect", "azimuth_polar"}
                    else "elevation axis (deg)"
                    if kind == "elevation"
                    else "frequency axis (GHz)"
                ),
                maximum_span=360.0 if kind == "azimuth_polar" else None,
            )
            y_limits, y_step = self._validated_fixed_axis(
                str(self.scale_mode_combo.currentData() or "shared_auto"),
                self.y_min_spin.value(),
                self.y_max_spin.value(),
                self.y_step_spin.value(),
                axis_name="vertical RCS axis",
            )
            return x_limits, x_step, y_limits, y_step

        @staticmethod
        def _axis_clipping_warnings(
            plots: Sequence[PlotSpec],
            *,
            x_limits: tuple[float, float] | None,
            x_step: float | None,
            y_limits: tuple[float, float] | None,
            y_step: float | None,
        ) -> tuple[str, ...]:
            warnings: list[str] = []

            def clipped_summary(
                values: Iterable[float],
                limits: tuple[float, float] | None,
                label: str,
            ) -> None:
                if limits is None:
                    return
                finite = tuple(float(value) for value in values if math.isfinite(value))
                if not finite:
                    return
                low, high = limits
                outside = sum(value < low or value > high for value in finite)
                if outside == 0:
                    return
                percent = 100.0 * outside / len(finite)
                if outside == len(finite):
                    warnings.append(
                        f"Fixed {label} limits contain no stored sample centers; "
                        "the rendered traces may be blank or show only crossing segments."
                    )
                else:
                    warnings.append(
                        f"Fixed {label} limits clip {outside} of {len(finite)} "
                        f"plotted samples ({percent:.1f}%)."
                    )

            clipped_summary(
                (
                    value
                    for plot in plots
                    for series in plot.series
                    for value in series.x
                ),
                x_limits,
                "horizontal-axis",
            )
            clipped_summary(
                (
                    value
                    for plot in plots
                    for series in plot.series
                    for value in series.y
                ),
                y_limits,
                "vertical-axis",
            )
            for label, limits, step in (
                ("horizontal", x_limits, x_step),
                ("vertical", y_limits, y_step),
            ):
                if limits is None or step is None:
                    continue
                count = int(math.floor((limits[1] - limits[0]) / step + 1.0e-9)) + 1
                if count > 40:
                    warnings.append(
                        f"The fixed {label} axis creates {count} major ticks; "
                        "increase the step for a readable slide."
                    )
            return tuple(warnings)

        def _finalize_plots(
            self,
            plots: Sequence[PlotSpec],
            entries: Sequence[DatasetCatalogEntry],
            *,
            x_limits: tuple[float, float] | None,
            x_tick_step: float | None,
            fixed_y_limits: tuple[float, float] | None,
            y_tick_step: float | None,
        ) -> tuple[PlotSpec, ...]:
            values = _with_shared_y_limits(plots, fixed_y_limits)
            values = self._apply_series_styles(values, entries)
            values = tuple(
                replace(
                    plot,
                    x_limits=x_limits,
                    x_tick_step=x_tick_step,
                    y_tick_step=y_tick_step,
                )
                for plot in values
            )
            self._last_plan_warnings = self._axis_clipping_warnings(
                values,
                x_limits=x_limits,
                x_step=x_tick_step,
                y_limits=fixed_y_limits,
                y_step=y_tick_step,
            )
            return values

        def _build_plan(self) -> PresentationPlan:
            entries = self._selected_entries()
            datasets = tuple(NamedGrid(entry.name, entry.grid) for entry in entries)
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
            if kind != "elevation" and elevation is None:
                raise ValueError("The selected datasets have no common elevation cut.")
            if polarization is None:
                raise ValueError("The selected datasets have no common polarization.")
            polarization_values = (
                ("VV", "HH")
                if str(polarization) == DUAL_COPOLARIZATION
                else (str(polarization),)
            )
            x_limits, x_tick_step, fixed_limits, y_tick_step = self._axis_overrides(
                kind
            )
            legend_mode = str(self.legend_mode_combo.currentData() or "master")
            show_plot_legends = legend_mode == "per_plot"
            show_master_legend = legend_mode == "master"
            deck_title = self.deck_title_edit.text().strip()
            if kind in {"azimuth_rect", "azimuth_polar", "elevation"}:
                frequencies = self.selected_frequencies()
                if not frequencies:
                    raise ValueError(
                        "Select one or more frequencies for the angular report."
                    )
                if len(frequencies) > _MAX_AZIMUTH_REPORT_FREQUENCIES:
                    raise ValueError(
                        "A single PPT report is limited to 60 angular-cut frequencies "
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
                if kind == "elevation":
                    azimuth = self.azimuth_combo.currentData()
                    if azimuth is None:
                        raise ValueError(
                            "The selected datasets have no common azimuth cut."
                        )
                    plots = build_elevation_specs(
                        datasets,
                        frequencies=frequencies,
                        azimuth=azimuth,
                        polarization=polarization,
                        quantity="magnitude",
                        angle_display_unit="deg",
                        frequency_display_unit="GHz",
                        y_limits=fixed_limits,
                        show_legend=show_plot_legends,
                    )
                    default_title = "RCS Elevation Sweeps"
                else:
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
                        show_legend=show_plot_legends,
                    )
                    default_title = "RCS Azimuth Sweeps"
                plots = self._finalize_plots(
                    plots,
                    entries,
                    x_limits=x_limits,
                    x_tick_step=x_tick_step,
                    fixed_y_limits=fixed_limits,
                    y_tick_step=y_tick_step,
                )
                return plan_azimuth_slides(
                    plots,
                    slide_titles=deck_title or default_title,
                    master_legend=show_master_legend,
                    polarization_labels=tuple(
                        value
                        for value in polarization_values
                        for _frequency in frequencies
                    ),
                )
            if kind == "frequency":
                azimuth_mode = str(
                    self.frequency_azimuth_mode_combo.currentData() or "exact"
                )
                azimuth = None
                azimuth_band = None
                azimuth_percentile = None
                if azimuth_mode == "exact":
                    azimuth = self.azimuth_combo.currentData()
                    if azimuth is None:
                        raise ValueError(
                            "The selected datasets have no common azimuth cut."
                        )
                elif azimuth_mode == "band":
                    if not self._azimuth_band_available:
                        raise ValueError(
                            "Azimuth-band percentile mode requires at least two "
                            "common azimuth samples across the selected datasets."
                        )
                    azimuth_band = (
                        float(self.azimuth_band_min_spin.value()),
                        float(self.azimuth_band_max_spin.value()),
                    )
                    azimuth_percentile = float(
                        self.azimuth_percentile_spin.value()
                    )
                else:
                    raise ValueError(
                        f"Unsupported frequency azimuth mode: {azimuth_mode!r}"
                    )

                plots = build_frequency_specs(
                    datasets,
                    azimuth=azimuth,
                    elevation=elevation,
                    polarization=polarization,
                    quantity="magnitude",
                    angle_display_unit="deg",
                    frequency_display_unit="GHz",
                    azimuth_band=azimuth_band,
                    azimuth_percentile=azimuth_percentile,
                    y_limits=fixed_limits,
                    show_legend=show_plot_legends,
                )
                plots = self._finalize_plots(
                    plots,
                    entries,
                    x_limits=x_limits,
                    x_tick_step=x_tick_step,
                    fixed_y_limits=fixed_limits,
                    y_tick_step=y_tick_step,
                )
                return plan_frequency_slides(
                    plots,
                    slide_titles=deck_title or "RCS Frequency Sweeps",
                    master_legend=show_master_legend,
                    polarization_labels=polarization_values,
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
            self._last_plan_warnings = ()
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
                self.axis_warning_label.clear()
                self.axis_warning_label.setVisible(False)
                self._update_navigation()
                return False
            self.export_button.setEnabled(True)
            self._last_error = ""
            message = (
                f"Preview ready: {len(plan.slides)} slide(s), "
                f"{plan.plot_count} plot(s). Review each page, choose an output, "
                "then export PPTX."
            )
            if self._last_plan_warnings:
                warning_text = "\n".join(
                    f"• {warning}" for warning in self._last_plan_warnings
                )
                self.axis_warning_label.setText(warning_text)
                self.axis_warning_label.setVisible(True)
                message += " Axis warning: " + " ".join(self._last_plan_warnings)
            else:
                self.axis_warning_label.clear()
                self.axis_warning_label.setVisible(False)
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
                    self._show_error("PowerPoint templates must use .pptx or .potx.")
                    return False
                if not template_path.is_file():
                    self._show_error(f"PowerPoint template not found: {template}")
                    return False
            template_layouts = self._selected_template_layouts()
            thread = QThread(self)
            # PresentationPlan and copied strings are immutable snapshots.
            # Subsequent shell/catalog/layout edits cannot change an export.
            worker = _ExportWorker(
                self._exporter,
                self._preview_plan,
                output,
                template,
                template_layouts,
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
            self.build_preview_button.setEnabled(not busy)
            self.export_button.setEnabled(not busy and self._preview_is_current)
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
                "Choose PowerPoint template",
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


def _physical_angle_sample_count(
    values: Sequence[float], unit: str, *, tol: float = 1.0e-6
) -> int:
    """Count distinct stored directions, collapsing the periodic seam alias."""

    unit_name = str(unit).strip().casefold()
    period = (
        2.0 * math.pi
        if unit_name in {"rad", "radian", "radians"}
        else 360.0
        if unit_name in {"deg", "degree", "degrees"}
        else None
    )
    unique: list[float] = []
    for raw_value in values:
        value = float(raw_value)
        if period is not None:
            value %= period
            if math.isclose(value, period, rel_tol=0.0, abs_tol=tol):
                value = 0.0
        if not any(
            math.isclose(value, existing, rel_tol=0.0, abs_tol=tol)
            for existing in unique
        ):
            unique.append(value)
    return len(unique)


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
