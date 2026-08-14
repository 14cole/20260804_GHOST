from __future__ import annotations

import base64
import os
import sys

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QByteArray, QMimeData, QTimer, Signal
from PySide6.QtGui import QColor, QDrag, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidgetItem,
    QListWidget,
    QMainWindow,
    QMenu,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QSplashScreen,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from assembly_tree import AssemblyTreePanel, MIME_BRANCH, MIME_DATASET
from grim_dataset import RcsGrid
from grim_cut_dataset_mixin import DatasetOpsMixin
from grim_cut_plot_mixin import PlotOpsMixin
from plot_models import PlotContext

BLUE_PALETTE = {
    "is_dark": True,
    "win_bg": "#0f172a",
    "panel_bg": "#0b1222",
    "text": "#dbeafe",
    "head_bg": "#172554",
    "border": "#1e3a8a",
    "hover": "#1d4ed8",
    "checked_bg": "#2563eb",
    "checked_border": "#3b82f6",
    "grid": "#475569",
    "fg": "#dbeafe",
}
SPLASH_DURATION_MS = 4000

# Plot-operation buttons, per tab: (row1_specs, row2_specs). Each spec is
# (button label, role key). Roles drive both the attribute wiring in
# _activate_plot_tab and the signal connections in __init__.
PLOT_OPS_SPECS = {
    "plotting": (
        (
            ("Hold", "hold"),
            ("Clear", "clear"),
            ("Azimuth (Rect)", "azimuth_rect"),
            ("Azimuth (Polar)", "azimuth_polar"),
            ("Frequency", "frequency"),
            ("Elevation Sweep", "elevation_sweep"),
            ("Waterfall", "waterfall"),
            ("Compare", "compare"),
        ),
        (
            ("Fit X", "fit_x"),
            ("Fit Y", "fit_y"),
            ("Fit Both", "fit_both"),
            ("Zoom Box", "zoom_box"),
            ("Pan", "pan"),
            ("Auto Plot", "auto_plot"),
            ("Auto Scale", "auto_scale"),
            ("PbP", "pbp"),
            ("Phase", "phase"),
        ),
    ),
    "isar": (
        (
            ("Hold", "hold"),
            ("Clear", "clear"),
            ("ISAR Image", "isar_image"),
            ("Az. vs D.R.", "az_vs_range"),
        ),
        (
            ("Fit X", "fit_x"),
            ("Fit Y", "fit_y"),
            ("Fit Both", "fit_both"),
            ("Zoom Box", "zoom_box"),
            ("Pan", "pan"),
            ("Auto Plot", "auto_plot"),
            ("Auto Scale", "auto_scale"),
        ),
    ),
}


def _branch_arrow_uri(points: str, fill: str) -> str:
    """Return a base64 SVG data-URI for a small polygon arrow (used in QSS branch rules)."""
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8">'
        f'<polygon points="{points}" fill="{fill}"/>'
        f'</svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def build_qss(palette: dict[str, str]) -> str:
    arrow_right = _branch_arrow_uri("2,1 6,4 2,7", palette["text"])   # collapsed
    arrow_down  = _branch_arrow_uri("1,2 7,2 4,6", palette["text"])   # expanded
    return f"""
    QMainWindow {{ background: {palette['win_bg']}; }}
    QFrame {{ background: {palette['panel_bg']}; border: 1px solid {palette['border']}; border-radius: 8px; }}
    QFrame#paramSeparator {{
        background: {palette['border']}; min-width: 2px; max-width: 2px; border: none; border-radius: 0px;
    }}
    QGroupBox {{ color: {palette['text']}; border: 1px solid {palette['border']}; border-radius: 8px; margin-top: 10px; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
    QLabel {{ color: {palette['text']}; }}
    QTableWidget {{
        background: {palette['panel_bg']}; color: {palette['text']};
        border: 1px solid {palette['border']}; gridline-color: {palette['grid']};
    }}
    QHeaderView::section {{ background: {palette['head_bg']}; color: {palette['text']}; border: none; padding: 6px; }}
    QTabWidget::pane {{ border: 1px solid {palette['border']}; background: {palette['panel_bg']}; }}
    QTabBar::tab {{ background: {palette['panel_bg']}; color: {palette['text']}; border: 1px solid {palette['border']}; border-bottom: 0; padding: 6px 12px; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px; }}
    QTabBar::tab:selected {{ background: {palette['head_bg']}; color: {palette['text']}; border-color: {palette['checked_border']}; }}
    QTabBar::tab:hover {{ background: {palette['hover']}; }}
    QListWidget {{ background: {palette['panel_bg']}; color: {palette['text']}; border: 1px solid {palette['border']}; }}
    QTreeWidget {{ background: {palette['panel_bg']}; color: {palette['text']}; border: 1px solid {palette['border']}; }}
    QTreeWidget::item {{ border-bottom: 1px solid {palette['grid']}; padding: 3px 4px; }}
    QTreeWidget::item:selected {{ background: {palette['checked_bg']}; color: white; }}
    QTreeWidget::branch {{ background: {palette['panel_bg']}; }}
    QTreeWidget::branch:has-children:!open {{ image: url("{arrow_right}"); }}
    QTreeWidget::branch:has-children:open  {{ image: url("{arrow_down}"); }}
    QTreeWidget#assemblyTree::branch:has-children {{ image: none; }}
    QListWidget::item {{ border-bottom: 1px solid {palette['grid']}; padding: 4px 6px; }}
    QListWidget QLineEdit {{
        background: {palette['panel_bg']}; color: {palette['text']}; border: 1px solid {palette['border']};
        padding: 2px 4px; min-height: 20px; font-size: 12px;
    }}
    QListWidget::item:selected {{
        background: {palette['checked_bg']}; color: white; border-bottom: 1px solid {palette['grid']};
    }}
    QToolButton, QDoubleSpinBox, QCheckBox, QLineEdit, QComboBox {{
        background: {palette['panel_bg']}; color: {palette['text']}; border: 1px solid {palette['border']};
        border-radius: 6px; padding: 6px;
    }}
    QCheckBox::indicator {{
        width: 14px; height: 14px;
        border: 1px solid {palette['border']};
        border-radius: 3px;
        background: {palette['panel_bg']};
    }}
    QCheckBox::indicator:checked {{
        background: {palette['checked_bg']};
        border-color: {palette['checked_border']};
    }}
    QToolButton:hover {{ border-color: {palette['hover']}; }}
    QToolButton:checked {{ background: {palette['checked_bg']}; color: white; border-color: {palette['checked_border']}; }}
    QComboBox QAbstractItemView {{ background: {palette['panel_bg']}; color: {palette['text']}; border: 1px solid {palette['border']}; }}
    QTableWidget::item:selected {{ background: {palette['checked_bg']}; color: white; }}
    QLabel#hoverReadout {{
        background: {palette['head_bg']}; color: {palette['text']}; border: 1px solid {palette['border']};
        border-radius: 4px; padding: 2px 6px; font-family: "Consolas","Courier New",monospace; font-size: 11px;
    }}
    QScrollArea#controlDock {{ background: {palette['win_bg']}; border: none; }}
    QWidget#dockBody {{ background: {palette['win_bg']}; }}
    QToolButton#sectionHeader {{
        background: {palette['head_bg']}; color: {palette['text']};
        border: 1px solid {palette['border']}; border-radius: 6px;
        padding: 7px 10px; text-align: left; font-weight: 600;
    }}
    QToolButton#sectionHeader:hover {{ border-color: {palette['hover']}; }}
    QToolButton#sectionHeader:checked {{ background: {palette['head_bg']}; color: {palette['text']}; border-color: {palette['border']}; }}
    QWidget#sectionBody {{
        background: {palette['panel_bg']}; border: 1px solid {palette['border']};
        border-top: none; border-top-left-radius: 0px; border-top-right-radius: 0px;
        border-bottom-left-radius: 6px; border-bottom-right-radius: 6px;
    }}
    QLabel#opsCategory {{ color: {palette['text']}; font-weight: 600; padding: 6px 2px 1px 2px; }}
    QLabel#paramHeader {{ color: {palette['text']}; font-weight: 600; padding: 2px; }}
    QLabel#plotTitle {{ color: {palette['text']}; font-weight: 700; font-size: 14px; padding: 2px 4px; }}
    QFrame#plotToolbar {{ background: {palette['head_bg']}; border: 1px solid {palette['border']}; border-radius: 8px; }}
    QFrame#datasetOpsPanel {{ background: {palette['panel_bg']}; border: 1px solid {palette['border']}; border-radius: 8px; }}
    """


def _extract_supported_drop_paths(mime: QMimeData) -> list[str]:
    if not mime.hasUrls():
        return []
    paths: list[str] = []
    for url in mime.urls():
        if not url.isLocalFile():
            continue
        path = url.toLocalFile()
        if path.lower().endswith((".grim", ".csv", ".txt", ".out", ".pio", ".cmplx_di", ".ss")):
            paths.append(path)
    return paths


class DatasetTable(QTableWidget):
    files_dropped = Signal(list)
    # branch_name: str, list of (name: str, grid: RcsGrid | None) tuples
    assembly_branch_dropped = Signal(str, list)
    rows_reordered = Signal()
    delete_requested = Signal()

    def __init__(self, rows: int, columns: int, parent: QWidget | None = None) -> None:
        super().__init__(rows, columns, parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self._pending_drag_data: tuple | None = None  # (name, RcsGrid|None)
        self._pending_drag_rows: list[int] = []

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.selectionModel().hasSelection():
            self.delete_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def startDrag(self, _) -> None:
        rows = sorted({item.row() for item in self.selectedItems()})
        if not rows:
            return
        entries = []
        for row in rows:
            name_item = self.item(row, 0)
            if name_item is not None:
                entries.append((name_item.text(), name_item.data(Qt.UserRole)))
        if not entries:
            return
        self._pending_drag_data = entries  # list of (name, RcsGrid|None)
        self._pending_drag_rows = rows
        mime = QMimeData()
        mime.setData(MIME_DATASET, QByteArray(entries[0][0].encode("utf-8")))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.CopyAction | Qt.MoveAction)
        self._pending_drag_data = None
        self._pending_drag_rows = []

    def dragEnterEvent(self, event) -> None:
        mime = event.mimeData()
        if event.source() is self:
            event.acceptProposedAction()
        elif mime.hasUrls() or mime.hasFormat(MIME_BRANCH):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        mime = event.mimeData()
        if event.source() is self:
            event.acceptProposedAction()
        elif mime.hasUrls() or mime.hasFormat(MIME_BRANCH):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        mime = event.mimeData()
        if event.source() is self and self._pending_drag_rows:
            self._reorder_to_drop(event)
            event.acceptProposedAction()
            return
        if mime.hasFormat(MIME_BRANCH):
            src = event.source()
            if hasattr(src, "_pending_branch_data") and src._pending_branch_data:
                branch_name = bytes(mime.data(MIME_BRANCH)).decode("utf-8")
                self.assembly_branch_dropped.emit(branch_name, src._pending_branch_data)
            event.acceptProposedAction()
        elif mime.hasUrls():
            paths = [u.toLocalFile() for u in mime.urls() if u.isLocalFile()]
            if paths:
                self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def _reorder_to_drop(self, event) -> None:
        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
        drop_index = self.indexAt(pos)
        if drop_index.isValid():
            target_row = drop_index.row()
            if self.dropIndicatorPosition() == QAbstractItemView.BelowItem:
                target_row += 1
        else:
            target_row = self.rowCount()

        src_rows = sorted(set(self._pending_drag_rows))
        if not src_rows:
            return
        # No-op if dropping onto the same contiguous range.
        if src_rows[0] <= target_row <= src_rows[-1] + 1 and src_rows == list(range(src_rows[0], src_rows[-1] + 1)):
            return

        col_count = self.columnCount()
        # Snapshot rows to move (items only; row indices change as we remove).
        snapshots: list[list[QTableWidgetItem | None]] = []
        for r in src_rows:
            snapshots.append([self.takeItem(r, c) for c in range(col_count)])

        # Remove source rows bottom-up; adjust target for rows removed above it.
        for r in reversed(src_rows):
            self.removeRow(r)
            if r < target_row:
                target_row -= 1

        # Insert at target in original order.
        for offset, row_items in enumerate(snapshots):
            insert_at = target_row + offset
            self.insertRow(insert_at)
            for c, item in enumerate(row_items):
                if item is not None:
                    self.setItem(insert_at, c, item)

        self.clearSelection()
        if snapshots:
            self.setCurrentCell(target_row, 0)
            selection = self.selectionModel()
            for offset in range(len(snapshots)):
                idx = self.model().index(target_row + offset, 0)
                selection.select(
                    idx,
                    selection.Select | selection.Rows,
                )
        self.rows_reordered.emit()


class ClickableLabel(QLabel):
    doubleClicked = Signal()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.doubleClicked.emit()
        else:
            super().mouseDoubleClickEvent(event)


class PlotSettingsPopup(QFrame):
    """Top-level popup frame for plot settings. Closing it via the title-bar
    untoggles the bound toggle button so the button state mirrors visibility.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._toggle_button: QToolButton | None = None
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle("Plot Settings")

    def set_toggle_button(self, button: QToolButton) -> None:
        self._toggle_button = button

    def closeEvent(self, event) -> None:
        if self._toggle_button is not None and self._toggle_button.isChecked():
            self._toggle_button.setChecked(False)
        super().closeEvent(event)


class CollapsibleSection(QWidget):
    """A titled panel whose body collapses when its header is clicked.

    Purely presentational — it organises the control dock into Datasets /
    Parameters / Operations / Plot Tools groups while holding the exact same
    widgets the app has always used.
    """

    def __init__(self, title: str, parent: QWidget | None = None, expanded: bool = True) -> None:
        super().__init__(parent)
        self._title = title
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.header = QToolButton()
        self.header.setObjectName("sectionHeader")
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.setCursor(Qt.PointingHandCursor)
        self.header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._body = QWidget()
        self._body.setObjectName("sectionBody")
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(8, 8, 8, 8)
        self._body_layout.setSpacing(6)

        outer.addWidget(self.header)
        outer.addWidget(self._body)

        self.header.toggled.connect(self._sync)
        self._sync(expanded)

    def _sync(self, on: bool) -> None:
        self.header.setText(("▾  " if on else "▸  ") + self._title)
        self._body.setVisible(on)

    def addWidget(self, widget, stretch: int = 0) -> None:
        self._body_layout.addWidget(widget, stretch)

    def addLayout(self, layout, stretch: int = 0) -> None:
        self._body_layout.addLayout(layout, stretch)


class GrimCutWindow(DatasetOpsMixin, PlotOpsMixin, QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.palette = BLUE_PALETTE

        self.setWindowTitle("GRIM Cut")
        self.resize(1550, 900)
        self._dock_width = 480

        right = QWidget()
        self.setCentralWidget(right)

        # ---------- Main panel ----------
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.main_tabs = QTabWidget()
        right_layout.addWidget(self.main_tabs, 1)
        self._tab_key_for_index: dict[int, str] = {}
        self._plot_splitters: dict[str, QSplitter] = {}
        self._plot_contexts: dict[str, PlotContext] = {}
        self._plot_controls_by_tab: dict[str, dict[str, QToolButton]] = {}
        self._active_plot_tab = "plotting"
        self._dataset_ops_visible = False

        # Hover readout is debounced — rapid mouse-moves coalesce into one
        # update so the per-event z lookup (O(N) for QuadMesh / 3D scatter)
        # doesn't run hundreds of times a second.
        self._pending_hover: tuple | None = None
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(30)
        self._hover_timer.timeout.connect(self._flush_hover)

        self.tab_simple_plots = QWidget()
        simple_layout = QVBoxLayout(self.tab_simple_plots)
        simple_layout.setContentsMargins(10, 10, 10, 10)
        simple_layout.setSpacing(0)

        plot_splitter = QSplitter(Qt.Horizontal)
        simple_layout.addWidget(plot_splitter, 1)
        self._plot_splitters["plotting"] = plot_splitter

        plot_panel = QWidget()
        dock = QScrollArea()
        dock.setObjectName("controlDock")
        dock.setWidgetResizable(True)
        dock.setFrameShape(QFrame.NoFrame)
        dock.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        dock.setMinimumWidth(360)
        plot_splitter.addWidget(dock)
        plot_splitter.addWidget(plot_panel)
        plot_splitter.setStretchFactor(0, 0)
        plot_splitter.setStretchFactor(1, 1)
        plot_splitter.setSizes([self._dock_width, 1550 - self._dock_width])

        self._plot_contexts["plotting"] = self._build_plot_left_context(plot_panel, "plotting")

        dock_body = QWidget()
        dock_body.setObjectName("dockBody")
        dock_layout = QVBoxLayout(dock_body)
        dock_layout.setContentsMargins(8, 8, 8, 8)
        dock_layout.setSpacing(8)

        # ---------- Datasets section (top, grows to fill the dock) ----------
        sec_datasets = CollapsibleSection("Datasets")
        sec_datasets.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        dataset_actions = QHBoxLayout()
        self.btn_dataset_save = QToolButton(text="Save")
        self.btn_dataset_save_all = QToolButton(text="Save All")
        self.btn_dataset_delete = QToolButton(text="Delete")
        dataset_actions.addWidget(self.btn_dataset_save)
        dataset_actions.addWidget(self.btn_dataset_save_all)
        dataset_actions.addWidget(self.btn_dataset_delete)
        dataset_actions.addStretch(1)
        sec_datasets.addLayout(dataset_actions)

        self.table = DatasetTable(0, 3)
        self.table.setHorizontalHeaderLabels(["Name", "File", "History"])
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(
            QAbstractItemView.DoubleClicked
            | QAbstractItemView.EditKeyPressed
            | QAbstractItemView.SelectedClicked
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.setMinimumHeight(160)
        sec_datasets.addWidget(self.table, 1)

        # ---------- Parameters section (single 4-column strip) ----------
        sec_params = CollapsibleSection("Parameters")
        params_grid = QGridLayout()
        params_grid.setHorizontalSpacing(10)
        params_grid.setVerticalSpacing(4)
        for col in range(4):
            params_grid.setColumnStretch(col, 1)
        self.list_pol = QListWidget()
        self.list_freq = QListWidget()
        self.list_elev = QListWidget()
        self.list_az = QListWidget()
        for widget in (self.list_pol, self.list_freq, self.list_elev, self.list_az):
            widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
            widget.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
            widget.setMinimumHeight(96)
        lbl_pol = ClickableLabel("Pol")
        lbl_freq = ClickableLabel("Freq (GHz)")
        lbl_elev = ClickableLabel("El (deg)")
        lbl_az = ClickableLabel("Az (deg)")
        for lbl in (lbl_pol, lbl_freq, lbl_elev, lbl_az):
            lbl.setObjectName("paramHeader")
        # One row of headers, one row of lists, four columns across.
        params_grid.addWidget(lbl_pol, 0, 0)
        params_grid.addWidget(lbl_freq, 0, 1)
        params_grid.addWidget(lbl_elev, 0, 2)
        params_grid.addWidget(lbl_az, 0, 3)
        params_grid.addWidget(self.list_pol, 1, 0)
        params_grid.addWidget(self.list_freq, 1, 1)
        params_grid.addWidget(self.list_elev, 1, 2)
        params_grid.addWidget(self.list_az, 1, 3)
        sec_params.addLayout(params_grid)

        # ---------- Dataset Operations (pop-out panel beside the table) ----------
        # Shared across plot tabs and toggled from each tab's "Dataset
        # Operations" button; docked next to the datasets table, left of the plot.
        self._dataset_ops_panel = QFrame()
        self._dataset_ops_panel.setObjectName("datasetOpsPanel")
        self._dataset_ops_panel.setMinimumWidth(220)
        self._dataset_ops_panel.setVisible(False)
        ops_panel_layout = QVBoxLayout(self._dataset_ops_panel)
        ops_panel_layout.setContentsMargins(8, 8, 8, 8)
        ops_panel_layout.setSpacing(6)
        ops_panel_title = QLabel("Dataset Operations")
        ops_panel_title.setObjectName("plotTitle")
        ops_panel_layout.addWidget(ops_panel_title)

        def _ops_pad(title: str, specs: tuple[tuple[str, str], ...], cols: int = 2) -> None:
            cap = QLabel(title)
            cap.setObjectName("opsCategory")
            ops_panel_layout.addWidget(cap)
            grid = QGridLayout()
            grid.setHorizontalSpacing(4)
            grid.setVerticalSpacing(4)
            for i, (label, attr) in enumerate(specs):
                btn = QToolButton(text=label)
                btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                setattr(self, attr, btn)
                grid.addWidget(btn, i // cols, i % cols)
            for c in range(cols):
                grid.setColumnStretch(c, 1)
            ops_panel_layout.addLayout(grid)

        _ops_pad("Combine", (
            ("Coherent +", "btn_coherent_add"),
            ("Coherent -", "btn_coherent_sub"),
            ("Coherent ÷", "btn_coherent_div"),
            ("Incoherent +", "btn_incoherent_add"),
            ("Incoherent -", "btn_incoherent_sub"),
            ("Δ dB", "btn_dbdiff"),
            ("Join", "btn_join"),
            ("Overlap", "btn_overlap"),
        ))
        _ops_pad("Transform", (
            ("Slice", "btn_slice"),
            ("Stats", "btn_stats"),
            ("Align", "btn_align"),
            ("Interpolate", "btn_interpolate"),
            ("Mirror", "btn_mirror"),
            ("Wrap", "btn_wrap"),
            ("Shift", "btn_shift"),
            ("Round", "btn_round"),
            ("Offset", "btn_offset"),
            ("Medianize", "btn_medianize"),
            ("Duplicate", "btn_duplicate"),
        ))
        _ops_pad("Geometry & Units", (
            ("El→Az360", "btn_el_to_az360"),
            ("Swap El/Az", "btn_swap_el_az"),
            ("→ dBke", "btn_to_dbke"),
            ("→ dBsm", "btn_to_dbsm"),
            ("Conic ↔ GC", "btn_conic_gc"),
            ("Wedge → Conic", "btn_wedge_to_conic"),
        ))

        dock_layout.addWidget(sec_datasets, 1)
        dock_layout.addWidget(sec_params)
        ops_panel_layout.addStretch(1)

        # Dock the shared Dataset Operations panel between the control dock and
        # the plot, so it appears right next to the datasets table when shown.
        plot_splitter.insertWidget(1, self._dataset_ops_panel)
        plot_splitter.setStretchFactor(0, 0)
        plot_splitter.setStretchFactor(1, 0)
        plot_splitter.setStretchFactor(2, 1)
        plot_splitter.setSizes(
            [self._dock_width, 260, 1550 - self._dock_width - 260]
        )

        dock.setWidget(dock_body)
        self._shared_right_panel = dock

        self.main_tabs.addTab(self.tab_simple_plots, "Plotting")
        self._tab_key_for_index[self.main_tabs.count() - 1] = "plotting"

        self.tab_isar = QWidget()
        isar_layout = QVBoxLayout(self.tab_isar)
        isar_layout.setContentsMargins(10, 10, 10, 10)
        isar_layout.setSpacing(0)

        isar_splitter = QSplitter(Qt.Horizontal)
        isar_layout.addWidget(isar_splitter, 1)
        self._plot_splitters["isar"] = isar_splitter

        isar_left_panel = QWidget()
        isar_splitter.addWidget(isar_left_panel)
        isar_splitter.setStretchFactor(0, 1)

        isar_context = self._build_plot_left_context(isar_left_panel, "isar")
        self._plot_contexts["isar"] = isar_context

        self.main_tabs.addTab(self.tab_isar, "ISAR")
        self._tab_key_for_index[self.main_tabs.count() - 1] = "isar"

        self.status = self.statusBar()
        self.status.showMessage("Ready")

        self.active_dataset: RcsGrid | None = None
        self._dataset_selection_order: list[int] = []
        self.last_plot_mode: str | None = None
        self.btn_phase = None
        self.btn_zoom_box = None
        self.btn_pan = None
        self.btn_auto_scale = None
        self.pbp_fill_mode = "gray"
        self.pbp_fill_gray = "#7a7a7a"
        self.pbp_heatmap_samples = 80

        self.setStyleSheet(build_qss(BLUE_PALETTE))
        self.table.files_dropped.connect(self._handle_files_dropped)
        self.table.assembly_branch_dropped.connect(self._on_assembly_branch_dropped)
        self.table.rows_reordered.connect(self._on_dataset_rows_reordered)
        for context in self._plot_contexts.values():
            context.assembly_tree_panel.files_to_load.connect(self._handle_files_dropped)
            context.assembly_tree_panel.platform_built.connect(self._on_platform_built)
        self.table.itemSelectionChanged.connect(self._on_dataset_selection_changed)
        self.table.customContextMenuRequested.connect(self._on_dataset_context_menu)
        self.table.horizontalHeader().sectionDoubleClicked.connect(self._on_dataset_header_double_clicked)
        for context in self._plot_contexts.values():
            context.plot_canvas.setContextMenuPolicy(Qt.CustomContextMenu)
            context.plot_canvas.customContextMenuRequested.connect(self._on_plot_context_menu)
            context.plot_canvas.mpl_connect("scroll_event", self._on_plot_scroll_zoom)
            context.plot_canvas.mpl_connect("button_press_event", self._on_plot_mouse_press)
            context.plot_canvas.mpl_connect("motion_notify_event", self._on_plot_mouse_move)
            context.plot_canvas.mpl_connect("button_release_event", self._on_plot_mouse_release)
        self.list_pol.itemSelectionChanged.connect(self._on_polarization_selection_changed)
        self.list_freq.itemSelectionChanged.connect(self._on_param_selection_changed)
        self.list_elev.itemSelectionChanged.connect(self._on_param_selection_changed)
        self.list_az.itemSelectionChanged.connect(self._on_param_selection_changed)
        self._connect_param_list(self.list_pol, "polarization")
        self._connect_param_list(self.list_freq, "frequency")
        self._connect_param_list(self.list_elev, "elevation")
        self._connect_param_list(self.list_az, "azimuth")
        lbl_pol.doubleClicked.connect(lambda: self.list_pol.selectAll())
        lbl_freq.doubleClicked.connect(lambda: self.list_freq.selectAll())
        lbl_elev.doubleClicked.connect(lambda: self.list_elev.selectAll())
        lbl_az.doubleClicked.connect(lambda: self.list_az.selectAll())

        for controls in self._plot_controls_by_tab.values():
            if "azimuth_rect" in controls:
                controls["azimuth_rect"].clicked.connect(self._plot_azimuth_rect)
            if "frequency" in controls:
                controls["frequency"].clicked.connect(self._plot_frequency)
            if "elevation_sweep" in controls:
                controls["elevation_sweep"].clicked.connect(self._plot_elevation_sweep)
            if "waterfall" in controls:
                controls["waterfall"].clicked.connect(self._plot_waterfall)
            if "compare" in controls:
                controls["compare"].clicked.connect(self._plot_compare)
            if "clear" in controls:
                controls["clear"].clicked.connect(self._clear_plot)
            if "fit_x" in controls:
                controls["fit_x"].clicked.connect(self._fit_x)
            if "fit_y" in controls:
                controls["fit_y"].clicked.connect(self._fit_y)
            if "pbp" in controls:
                controls["pbp"].toggled.connect(self._on_pbp_toggled)
            if "azimuth_polar" in controls:
                controls["azimuth_polar"].clicked.connect(self._plot_azimuth_polar)
            if "isar_image" in controls:
                controls["isar_image"].clicked.connect(self._plot_isar_image)
            if "az_vs_range" in controls:
                controls["az_vs_range"].clicked.connect(self._plot_az_vs_range)
            if "fit_both" in controls:
                controls["fit_both"].clicked.connect(self._fit_both)
            if "phase" in controls:
                controls["phase"].toggled.connect(self._on_phase_toggled)
            if "zoom_box" in controls:
                controls["zoom_box"].toggled.connect(self._on_zoom_box_toggled)
            if "pan" in controls:
                controls["pan"].toggled.connect(self._on_pan_toggled)
            if "auto_scale" in controls:
                controls["auto_scale"].toggled.connect(self._on_auto_scale_toggled)

        self.btn_coherent_add.clicked.connect(self._coherent_add_selected)
        self.btn_coherent_sub.clicked.connect(self._coherent_sub_selected)
        self.btn_coherent_div.clicked.connect(self._coherent_div_selected)
        self.btn_incoherent_add.clicked.connect(self._incoherent_add_selected)
        self.btn_incoherent_sub.clicked.connect(self._incoherent_sub_selected)
        self.btn_dbdiff.clicked.connect(self._dbdiff_selected)
        self.btn_slice.clicked.connect(self._slice_selected)
        self.btn_stats.clicked.connect(self._statistics_selected)
        self.btn_join.clicked.connect(self._join_selected_datasets)
        self.btn_overlap.clicked.connect(self._overlap_selected_datasets)
        self.btn_align.clicked.connect(self._align_selected)
        self.btn_interpolate.clicked.connect(self._interpolate_selected)
        self.btn_mirror.clicked.connect(self._mirror_selected)
        self.btn_wrap.clicked.connect(self._wrap_selected)
        self.btn_shift.clicked.connect(self._shift_selected)
        self.btn_round.clicked.connect(self._round_selected)
        self.btn_offset.clicked.connect(self._offset_selected)
        self.btn_medianize.clicked.connect(self._medianize_selected)
        self.btn_duplicate.clicked.connect(self._duplicate_selected)
        self.btn_el_to_az360.clicked.connect(self._elevation_to_azimuth_360_selected)
        self.btn_swap_el_az.clicked.connect(self._swap_elevation_azimuth_selected)
        self.btn_to_dbke.clicked.connect(self._convert_to_dbke_selected)
        self.btn_to_dbsm.clicked.connect(self._convert_to_dbsm_selected)
        self.btn_conic_gc.clicked.connect(self._convert_conic_gc_selected)
        self.btn_wedge_to_conic.clicked.connect(self._convert_wedge_to_conic_selected)
        self.btn_dataset_save.clicked.connect(self._save_selected_datasets)
        self.btn_dataset_save_all.clicked.connect(self._save_all_datasets)
        self.btn_dataset_delete.clicked.connect(self._delete_selected_datasets)
        self.table.delete_requested.connect(self._delete_selected_datasets)

        # Window-scoped keyboard shortcuts for the most common dataset ops.
        # Ctrl++ also bound to Ctrl+= so users don't have to hold shift on US layouts.
        shortcut_specs = (
            ("Ctrl+J", self._join_selected_datasets),
            ("Ctrl+O", self._overlap_selected_datasets),
            ("Ctrl+-", self._coherent_sub_selected),
            ("Ctrl++", self._coherent_add_selected),
            ("Ctrl+=", self._coherent_add_selected),
            ("Ctrl+S", self._save_selected_datasets),
        )
        for key_seq, slot in shortcut_specs:
            sc = QShortcut(QKeySequence(key_seq), self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(slot)
        for tab_key, context in self._plot_contexts.items():
            context.btn_assembly_tree.toggled.connect(context.assembly_tree_panel.setVisible)
            context.btn_dataset_ops.toggled.connect(self._toggle_dataset_ops)
            context.btn_settings.toggled.connect(context.settings_frame.setVisible)
            context.btn_export_plot.clicked.connect(self._export_plot)
            context.chk_plot_legend.toggled.connect(self._update_legend_visibility)
            context.btn_plot_bg.clicked.connect(lambda _=False, which="bg": self._choose_plot_color(which))
            context.btn_plot_grid.clicked.connect(
                lambda _=False, which="grid": self._choose_plot_color(which)
            )
            context.btn_plot_text.clicked.connect(
                lambda _=False, which="text": self._choose_plot_color(which)
            )
            context.combo_polar_zero.currentIndexChanged.connect(self._on_polar_zero_changed)
            context.btn_isar_apply.clicked.connect(self._on_isar_window_changed)
            # On the ISAR tab every settings-frame change re-runs the FFT
            # imaging, which is too slow for per-keystroke spinbox updates.
            # Defer everything through the Apply button instead — EXCEPT the
            # color-scale spinboxes, which only retune the existing image's
            # clim (no recompute) and so can stay live.
            # The plotting tab keeps live updates because its plots are cheap.
            # Aperture scrubbing and peak-scaling stay live on every tab —
            # stepping/playing is the point of the scrub workflow, and the
            # async compute path coalesces bursts of requests.
            context.btn_isar_ap_prev.clicked.connect(self._on_isar_ap_prev)
            context.btn_isar_ap_next.clicked.connect(self._on_isar_ap_next)
            context.btn_isar_ap_play.toggled.connect(self._on_isar_ap_play)
            context.btn_isar_peak_scale.clicked.connect(self._on_isar_peak_scale)
            if tab_key == "isar":
                context.spin_plot_zmin.valueChanged.connect(self._on_isar_clim_changed)
                context.spin_plot_zmax.valueChanged.connect(self._on_isar_clim_changed)
                context.spin_plot_zstep.valueChanged.connect(self._on_isar_clim_changed)
                context.chk_isar_aperture.toggled.connect(self._on_isar_window_changed)
                context.spin_isar_ap_center.valueChanged.connect(self._on_isar_window_changed)
                context.spin_isar_ap_width.valueChanged.connect(self._on_isar_window_changed)
                continue
            context.spin_plot_xmin.valueChanged.connect(self._apply_plot_limits)
            context.spin_plot_xmax.valueChanged.connect(self._apply_plot_limits)
            context.spin_plot_ymin.valueChanged.connect(self._apply_plot_limits)
            context.spin_plot_ymax.valueChanged.connect(self._apply_plot_limits)
            context.spin_plot_xstep.valueChanged.connect(self._apply_plot_limits)
            context.spin_plot_ystep.valueChanged.connect(self._apply_plot_limits)
            context.spin_plot_zmin.valueChanged.connect(self._on_waterfall_style_changed)
            context.spin_plot_zmax.valueChanged.connect(self._on_waterfall_style_changed)
            context.spin_plot_zstep.valueChanged.connect(self._on_waterfall_style_changed)
            context.combo_plot_scale.currentIndexChanged.connect(self._on_plot_scale_changed)
            context.combo_colormap.currentTextChanged.connect(self._on_colormap_changed)
            context.chk_colorbar.toggled.connect(self._on_waterfall_style_changed)
            context.chk_colorbar_shared.toggled.connect(self._on_waterfall_style_changed)
            context.chk_plot_grid_visible.toggled.connect(self._apply_plot_theme)
            context.chk_colormap_invert.toggled.connect(self._on_colormap_changed)
            context.combo_isar_window.currentIndexChanged.connect(self._on_isar_window_changed)
            context.combo_isar_units.currentIndexChanged.connect(self._on_isar_window_changed)
            context.chk_isar_az_interp.toggled.connect(self._on_isar_window_changed)
            context.spin_isar_az_min.valueChanged.connect(self._on_isar_window_changed)
            context.spin_isar_az_max.valueChanged.connect(self._on_isar_window_changed)
            context.spin_isar_az_step.valueChanged.connect(self._on_isar_window_changed)
            context.chk_isar_freq_band.toggled.connect(self._on_isar_window_changed)
            context.spin_isar_freq_min.valueChanged.connect(self._on_isar_window_changed)
            context.spin_isar_freq_max.valueChanged.connect(self._on_isar_window_changed)
            context.combo_isar_recon.currentIndexChanged.connect(self._on_isar_window_changed)
            context.spin_isar_l1_strength.valueChanged.connect(self._on_isar_window_changed)
            context.spin_isar_l1_iters.valueChanged.connect(self._on_isar_window_changed)
            context.chk_isar_flip_x.toggled.connect(self._on_isar_window_changed)
            context.chk_isar_flip_y.toggled.connect(self._on_isar_window_changed)
            context.chk_isar_aperture.toggled.connect(self._on_isar_window_changed)
            context.spin_isar_ap_center.valueChanged.connect(self._on_isar_window_changed)
            context.spin_isar_ap_width.valueChanged.connect(self._on_isar_window_changed)
            context.chk_isar_square.toggled.connect(self._on_isar_window_changed)

        self._activate_plot_tab("plotting")
        self.main_tabs.currentChanged.connect(self._on_main_tab_changed)
        self._update_plot_color_buttons()

    def dragEnterEvent(self, event) -> None:
        if _extract_supported_drop_paths(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if _extract_supported_drop_paths(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        paths = _extract_supported_drop_paths(event.mimeData())
        if paths:
            self._handle_files_dropped(paths)
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def _build_plot_left_context(self, panel: QWidget, tab_key: str) -> PlotContext:
        left_layout = QVBoxLayout(panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        topbar = QHBoxLayout()
        plot_title = QLabel("Display")
        plot_title.setObjectName("plotTitle")
        topbar.addWidget(plot_title)
        topbar.addStretch(1)
        btn_assembly_tree = QToolButton(text="Assembly Tree")
        btn_assembly_tree.setCheckable(True)
        btn_dataset_ops = QToolButton(text="Dataset Operations")
        btn_dataset_ops.setCheckable(True)
        btn_export_plot = QToolButton(text="Export Plot")
        btn_settings = QToolButton(text="Plot Settings")
        btn_settings.setCheckable(True)
        topbar.addWidget(btn_assembly_tree)
        topbar.addWidget(btn_dataset_ops)
        topbar.addWidget(btn_export_plot)
        topbar.addWidget(btn_settings)
        left_layout.addLayout(topbar)

        settings_frame = PlotSettingsPopup(panel)
        settings_frame.setFrameShape(QFrame.StyledPanel)
        settings_frame.setVisible(False)
        settings_frame.set_toggle_button(btn_settings)
        settings_layout = QGridLayout(settings_frame)
        settings_layout.setHorizontalSpacing(8)
        settings_layout.setVerticalSpacing(6)
        settings_layout.setColumnStretch(1, 1)
        settings_layout.setColumnStretch(3, 1)
        settings_layout.setColumnStretch(5, 1)

        row = 0
        settings_layout.addWidget(QLabel("Plot X Min"), row, 0)
        spin_plot_xmin = QDoubleSpinBox()
        spin_plot_xmin.setRange(-1e9, 1e9)
        spin_plot_xmin.setValue(-180.0)
        settings_layout.addWidget(spin_plot_xmin, row, 1)
        settings_layout.addWidget(QLabel("Plot X Max"), row, 2)
        spin_plot_xmax = QDoubleSpinBox()
        spin_plot_xmax.setRange(-1e9, 1e9)
        spin_plot_xmax.setValue(180.0)
        settings_layout.addWidget(spin_plot_xmax, row, 3)
        settings_layout.addWidget(QLabel("Plot X Step"), row, 4)
        spin_plot_xstep = QDoubleSpinBox()
        spin_plot_xstep.setRange(0.0, 1e9)
        spin_plot_xstep.setDecimals(6)
        spin_plot_xstep.setValue(0.0)
        settings_layout.addWidget(spin_plot_xstep, row, 5)
        row += 1

        settings_layout.addWidget(QLabel("Plot Y Min"), row, 0)
        spin_plot_ymin = QDoubleSpinBox()
        spin_plot_ymin.setRange(-1e9, 1e9)
        spin_plot_ymin.setValue(-80.0)
        settings_layout.addWidget(spin_plot_ymin, row, 1)
        settings_layout.addWidget(QLabel("Plot Y Max"), row, 2)
        spin_plot_ymax = QDoubleSpinBox()
        spin_plot_ymax.setRange(-1e9, 1e9)
        spin_plot_ymax.setValue(0.0)
        settings_layout.addWidget(spin_plot_ymax, row, 3)
        settings_layout.addWidget(QLabel("Plot Y Step"), row, 4)
        spin_plot_ystep = QDoubleSpinBox()
        spin_plot_ystep.setRange(0.0, 1e9)
        spin_plot_ystep.setDecimals(6)
        spin_plot_ystep.setValue(0.0)
        settings_layout.addWidget(spin_plot_ystep, row, 5)
        row += 1

        settings_layout.addWidget(QLabel("Plot Z Min"), row, 0)
        spin_plot_zmin = QDoubleSpinBox()
        spin_plot_zmin.setRange(-1e9, 1e9)
        spin_plot_zmin.setValue(0.0)
        settings_layout.addWidget(spin_plot_zmin, row, 1)
        settings_layout.addWidget(QLabel("Plot Z Max"), row, 2)
        spin_plot_zmax = QDoubleSpinBox()
        spin_plot_zmax.setRange(-1e9, 1e9)
        spin_plot_zmax.setValue(0.0)
        settings_layout.addWidget(spin_plot_zmax, row, 3)
        settings_layout.addWidget(QLabel("Plot Z Step"), row, 4)
        spin_plot_zstep = QDoubleSpinBox()
        spin_plot_zstep.setRange(0.0, 1e9)
        spin_plot_zstep.setDecimals(6)
        spin_plot_zstep.setValue(0.0)
        settings_layout.addWidget(spin_plot_zstep, row, 5)
        row += 1

        settings_layout.addWidget(QLabel("Plot Scale"), row, 0)
        combo_plot_scale = QComboBox()
        combo_plot_scale.addItem("dBsm", "dbsm")
        combo_plot_scale.addItem("Linear", "linear")
        default_index = combo_plot_scale.findData("dbsm")
        if default_index >= 0:
            combo_plot_scale.setCurrentIndex(default_index)
        settings_layout.addWidget(combo_plot_scale, row, 1, 1, 5)
        row += 1

        settings_layout.addWidget(QLabel("Polar 0° Direction"), row, 0)
        combo_polar_zero = QComboBox()
        polar_zero_options = [
            ("North", "N"),
            ("North East", "NE"),
            ("East", "E"),
            ("South East", "SE"),
            ("South", "S"),
            ("South West", "SW"),
            ("West", "W"),
            ("North West", "NW"),
        ]
        for label, loc in polar_zero_options:
            combo_polar_zero.addItem(label, loc)
        default_index = combo_polar_zero.findData("N")
        if default_index >= 0:
            combo_polar_zero.setCurrentIndex(default_index)
        settings_layout.addWidget(combo_polar_zero, row, 1, 1, 5)
        row += 1

        settings_layout.addWidget(QLabel("Colormap"), row, 0)
        combo_colormap = QComboBox()
        combo_colormap.addItems(
            ["viridis", "plasma", "inferno", "magma", "cividis", "turbo"]
        )
        settings_layout.addWidget(combo_colormap, row, 1)
        chk_colorbar = QCheckBox("Show Colorbar")
        chk_colorbar.setChecked(True)
        settings_layout.addWidget(chk_colorbar, row, 2)
        chk_colorbar_shared = QCheckBox("Shared Colorbar")
        chk_colorbar_shared.setChecked(True)
        settings_layout.addWidget(chk_colorbar_shared, row, 3)
        row += 1

        settings_layout.addWidget(QLabel("ISAR Window"), row, 0)
        combo_isar_window = QComboBox()
        combo_isar_window.addItems([
            "Hanning",
            "Hamming",
            "Blackman",
            "Blackman-Harris",
            "Kaiser β=15",
            "Rectangular",
        ])
        settings_layout.addWidget(combo_isar_window, row, 1)
        settings_layout.addWidget(QLabel("ISAR Units"), row, 2)
        combo_isar_units = QComboBox()
        combo_isar_units.addItems(["m", "in", "ft"])
        settings_layout.addWidget(combo_isar_units, row, 3)
        row += 1

        chk_isar_az_interp = QCheckBox("Interp Az")
        chk_isar_az_interp.setToolTip(
            "Resample azimuth onto a uniform grid before imaging. Periodic "
            "(360°-wrapping) when the source covers ≥359°; otherwise linear "
            "with zero-fill outside the source support."
        )
        settings_layout.addWidget(chk_isar_az_interp, row, 0)
        spin_isar_az_min = QDoubleSpinBox()
        spin_isar_az_min.setRange(-3600.0, 3600.0)
        spin_isar_az_min.setDecimals(4)
        spin_isar_az_min.setSingleStep(1.0)
        spin_isar_az_min.setValue(0.0)
        spin_isar_az_min.setToolTip("Lower azimuth limit (deg) for the uniform target grid.")
        settings_layout.addWidget(spin_isar_az_min, row, 1)
        spin_isar_az_max = QDoubleSpinBox()
        spin_isar_az_max.setRange(-3600.0, 3600.0)
        spin_isar_az_max.setDecimals(4)
        spin_isar_az_max.setSingleStep(1.0)
        spin_isar_az_max.setValue(360.0)
        spin_isar_az_max.setToolTip("Upper azimuth limit (deg) for the uniform target grid.")
        settings_layout.addWidget(spin_isar_az_max, row, 2)
        spin_isar_az_step = QDoubleSpinBox()
        spin_isar_az_step.setRange(1.0e-4, 90.0)
        spin_isar_az_step.setDecimals(4)
        spin_isar_az_step.setSingleStep(0.1)
        spin_isar_az_step.setValue(1.0)
        spin_isar_az_step.setToolTip("Azimuth step (deg) for the uniform target grid.")
        settings_layout.addWidget(spin_isar_az_step, row, 3)
        row += 1

        chk_isar_freq_band = QCheckBox("Freq Band")
        chk_isar_freq_band.setToolTip(
            "Limit ISAR imaging to selected frequencies within [min, max] "
            "(dataset frequency units, typically GHz). Lets you sweep sub-bands "
            "numerically instead of re-selecting thousands of list entries."
        )
        settings_layout.addWidget(chk_isar_freq_band, row, 0)
        spin_isar_freq_min = QDoubleSpinBox()
        spin_isar_freq_min.setRange(0.0, 10000.0)
        spin_isar_freq_min.setDecimals(4)
        spin_isar_freq_min.setSingleStep(0.5)
        spin_isar_freq_min.setValue(0.0)
        spin_isar_freq_min.setToolTip("Lower frequency limit for ISAR imaging.")
        settings_layout.addWidget(spin_isar_freq_min, row, 1)
        spin_isar_freq_max = QDoubleSpinBox()
        spin_isar_freq_max.setRange(0.0, 10000.0)
        spin_isar_freq_max.setDecimals(4)
        spin_isar_freq_max.setSingleStep(0.5)
        spin_isar_freq_max.setValue(18.0)
        spin_isar_freq_max.setToolTip("Upper frequency limit for ISAR imaging.")
        settings_layout.addWidget(spin_isar_freq_max, row, 2)
        row += 1

        settings_layout.addWidget(QLabel("Recon"), row, 0)
        combo_isar_recon = QComboBox()
        combo_isar_recon.addItems(["FFT (fast)", "Sparse L1 (clean)"])
        combo_isar_recon.setToolTip(
            "FFT: matched-filter imaging — instant, but point-spread sidelobes "
            "smear scatterers into crosses and haze.\n"
            "Sparse L1: basis-pursuit-denoise reconstruction (van den Berg & "
            "Friedlander 2008) — solves for the fewest scatterers that explain "
            "the data, so sidelobes vanish and the object outline stands out. "
            "Slower (iterative); intended for sub-aperture / sub-band looks. "
            "The taper window only applies to FFT mode."
        )
        settings_layout.addWidget(combo_isar_recon, row, 1)
        spin_isar_l1_strength = QDoubleSpinBox()
        spin_isar_l1_strength.setRange(0.001, 0.9)
        spin_isar_l1_strength.setDecimals(3)
        spin_isar_l1_strength.setSingleStep(0.01)
        spin_isar_l1_strength.setValue(0.05)
        spin_isar_l1_strength.setToolTip(
            "Sparsity strength (λ as a fraction of the matched-filter peak). "
            "Higher = cleaner/sparser image but faint scatterers drop out; "
            "lower keeps weak features at the cost of residual haze."
        )
        settings_layout.addWidget(spin_isar_l1_strength, row, 2)
        spin_isar_l1_iters = QDoubleSpinBox()
        spin_isar_l1_iters.setRange(10, 1000)
        spin_isar_l1_iters.setDecimals(0)
        spin_isar_l1_iters.setSingleStep(25)
        spin_isar_l1_iters.setValue(100)
        spin_isar_l1_iters.setSuffix(" it")
        spin_isar_l1_iters.setToolTip(
            "Sparse solver iterations. 100 is usually converged; raise it if "
            "the image still changes between runs."
        )
        settings_layout.addWidget(spin_isar_l1_iters, row, 3)
        chk_isar_flip_x = QCheckBox("Flip X")
        chk_isar_flip_x.setToolTip(
            "Mirror the image about x=0 (swap left/right). Use when the "
            "cross-range axis comes out reversed versus another tool — an "
            "opposite azimuth rotation-direction convention. Validate once "
            "against a known asymmetric geometry, then leave it set."
        )
        settings_layout.addWidget(chk_isar_flip_x, row, 4)
        chk_isar_flip_y = QCheckBox("Flip Y")
        chk_isar_flip_y.setToolTip(
            "Mirror the image about y=0 (swap up/down). Use when the range "
            "axis comes out reversed versus another tool — an opposite "
            "down-range sign convention. Both flips together are equivalent "
            "to the opposite e^{+j2kr} phase convention."
        )
        settings_layout.addWidget(chk_isar_flip_y, row, 5)
        row += 1

        chk_isar_aperture = QCheckBox("Aperture")
        chk_isar_aperture.setToolTip(
            "Scrub mode: image only the selected azimuths within ±width/2 of the "
            "center look angle (wraps at 0/360°). Drive the center numerically or "
            "with the ◀ ▶ step buttons (half-aperture steps) — changes render "
            "live, no Apply needed."
        )
        settings_layout.addWidget(chk_isar_aperture, row, 0)
        spin_isar_ap_center = QDoubleSpinBox()
        spin_isar_ap_center.setRange(0.0, 360.0)
        spin_isar_ap_center.setDecimals(3)
        spin_isar_ap_center.setSingleStep(1.0)
        spin_isar_ap_center.setValue(0.0)
        spin_isar_ap_center.setWrapping(True)
        spin_isar_ap_center.setToolTip("Aperture center look angle (deg).")
        settings_layout.addWidget(spin_isar_ap_center, row, 1)
        spin_isar_ap_width = QDoubleSpinBox()
        spin_isar_ap_width.setRange(0.01, 360.0)
        spin_isar_ap_width.setDecimals(3)
        spin_isar_ap_width.setSingleStep(1.0)
        spin_isar_ap_width.setValue(5.0)
        spin_isar_ap_width.setToolTip("Aperture width (deg).")
        settings_layout.addWidget(spin_isar_ap_width, row, 2)
        btn_isar_ap_prev = QToolButton(text="◀")
        btn_isar_ap_prev.setToolTip("Step the aperture center back by half a width.")
        settings_layout.addWidget(btn_isar_ap_prev, row, 3)
        btn_isar_ap_next = QToolButton(text="▶")
        btn_isar_ap_next.setToolTip("Step the aperture center forward by half a width.")
        settings_layout.addWidget(btn_isar_ap_next, row, 4)
        btn_isar_ap_play = QToolButton(text="Play")
        btn_isar_ap_play.setCheckable(True)
        btn_isar_ap_play.setToolTip(
            "Auto-step the aperture around the target, rendering each look as "
            "fast as it computes. Click again to stop."
        )
        settings_layout.addWidget(btn_isar_ap_play, row, 5)
        row += 1

        btn_isar_peak_scale = QToolButton(text="Peak −")
        btn_isar_peak_scale.setToolTip(
            "Set the color scale to [peak − N, peak] of the current ISAR image — "
            "the standard dynamic-range convention. Instant (no recompute)."
        )
        settings_layout.addWidget(btn_isar_peak_scale, row, 0)
        spin_isar_peak_drop = QDoubleSpinBox()
        spin_isar_peak_drop.setRange(1.0, 200.0)
        spin_isar_peak_drop.setDecimals(1)
        spin_isar_peak_drop.setSingleStep(5.0)
        spin_isar_peak_drop.setValue(40.0)
        spin_isar_peak_drop.setSuffix(" dB")
        spin_isar_peak_drop.setToolTip("Dynamic range below the image peak.")
        settings_layout.addWidget(spin_isar_peak_drop, row, 1)
        row += 1

        chk_isar_square = QCheckBox("Square Aspect")
        chk_isar_square.setToolTip(
            "Lock the image to equal cross-range / down-range scale and clip the "
            "visible window to a square centred on (0, 0). The square is sized to "
            "the smaller of the down-range / cross-range half-extents so the target "
            "fills the box and the geometry is undistorted. Off uses 'fill the axes' "
            "scaling, which packs more data on screen but stretches the geometry."
        )
        settings_layout.addWidget(chk_isar_square, row, 0, 1, 2)
        btn_isar_apply = QToolButton(text="Apply ISAR Settings")
        btn_isar_apply.setToolTip(
            "Re-render the ISAR image with the current settings. On the ISAR tab, "
            "settings changes are deferred until you click here so typing into "
            "spinboxes doesn't trigger a re-image per keystroke."
        )
        settings_layout.addWidget(btn_isar_apply, row, 2, 1, 4)
        row += 1

        chk_plot_grid_visible = QCheckBox("Show Grid")
        chk_plot_grid_visible.setChecked(True)
        settings_layout.addWidget(chk_plot_grid_visible, row, 0)
        chk_colormap_invert = QCheckBox("Invert Colormap")
        chk_colormap_invert.setChecked(False)
        settings_layout.addWidget(chk_colormap_invert, row, 1)
        row += 1

        settings_layout.addWidget(QLabel("Plot Colors"), row, 0)
        btn_plot_bg = QToolButton(text="BG")
        btn_plot_grid = QToolButton(text="Grid")
        btn_plot_text = QToolButton(text="Text")
        settings_layout.addWidget(btn_plot_bg, row, 1)
        settings_layout.addWidget(btn_plot_grid, row, 2)
        settings_layout.addWidget(btn_plot_text, row, 3)

        plot_frame = QFrame()
        plot_frame.setFrameShape(QFrame.StyledPanel)
        plot_layout = QVBoxLayout(plot_frame)
        plot_layout.setContentsMargins(20, 20, 20, 20)
        plot_layout.setSpacing(12)

        plot_figure = Figure(facecolor=BLUE_PALETTE["panel_bg"])
        plot_canvas = FigureCanvas(plot_figure)
        plot_canvas.setMinimumSize(320, 240)
        plot_canvas.setStyleSheet("background: transparent;")
        plot_ax = plot_figure.add_subplot(111)
        plot_ax.set_facecolor(BLUE_PALETTE["panel_bg"])
        plot_ax.grid(True, color=BLUE_PALETTE["grid"], alpha=0.35)
        plot_ax.tick_params(colors=BLUE_PALETTE["text"])
        plot_ax.xaxis.label.set_color(BLUE_PALETTE["text"])
        plot_ax.yaxis.label.set_color(BLUE_PALETTE["text"])
        for spine in plot_ax.spines.values():
            spine.set_color(BLUE_PALETTE["border"])
        plot_canvas.draw_idle()
        plot_layout.addWidget(plot_canvas, 1)
        hover_readout = QLabel("x: --   y: --")
        hover_readout.setObjectName("hoverReadout")
        hover_readout.setTextInteractionFlags(Qt.TextSelectableByMouse)
        plot_layout.addWidget(hover_readout, 0, Qt.AlignLeft)
        plot_canvas.mpl_connect(
            "motion_notify_event",
            lambda event, lbl=hover_readout: self._schedule_hover(event, lbl),
        )
        plot_canvas.mpl_connect(
            "axes_leave_event",
            lambda event, lbl=hover_readout: self._reset_hover_readout(lbl),
        )
        plot_canvas.mpl_connect(
            "figure_leave_event",
            lambda event, lbl=hover_readout: self._reset_hover_readout(lbl),
        )

        # Plot-operations toolbar — docked above the plot, actions split across
        # two rows so the toolbar's minimum width stays narrow (otherwise a
        # single long row forces the whole plot area wider than the screen and
        # pushes the right-hand buttons off-screen when the side panels open).
        row1_specs, row2_specs = PLOT_OPS_SPECS[tab_key]
        plot_controls: dict[str, QToolButton] = {}

        def _make_plot_button(label: str, role: str) -> QToolButton:
            btn = QToolButton(text=label)
            if role in ("hold", "auto_plot", "auto_scale", "pbp", "phase", "zoom_box", "pan"):
                btn.setCheckable(True)
            plot_controls[role] = btn
            return btn

        # Legend toggle sits at the head of the toolbar (left of Hold); it
        # replaces the old "Show Legend" checkbox in the Plot Settings window.
        chk_plot_legend = QToolButton(text="Legend")
        chk_plot_legend.setCheckable(True)
        chk_plot_legend.setChecked(True)
        chk_plot_legend.setToolTip("Show or hide the plot legend")

        plot_ops_bar = QFrame()
        plot_ops_bar.setObjectName("plotToolbar")
        plot_ops_bar_layout = QVBoxLayout(plot_ops_bar)
        plot_ops_bar_layout.setContentsMargins(8, 6, 8, 6)
        plot_ops_bar_layout.setSpacing(4)
        for _row_index, _specs in enumerate((row1_specs, row2_specs)):
            bar_row = QHBoxLayout()
            bar_row.setSpacing(4)
            if _row_index == 0:
                bar_row.addWidget(chk_plot_legend)
            for label, role in _specs:
                bar_row.addWidget(_make_plot_button(label, role))
            bar_row.addStretch(1)
            plot_ops_bar_layout.addLayout(bar_row)
        self._plot_controls_by_tab[tab_key] = plot_controls

        assembly_tree_panel = AssemblyTreePanel()
        assembly_tree_panel.setVisible(False)

        inner_split = QSplitter(Qt.Horizontal)
        inner_split.addWidget(assembly_tree_panel)
        inner_split.addWidget(plot_frame)
        inner_split.setStretchFactor(0, 0)
        inner_split.setStretchFactor(1, 1)
        inner_split.setSizes([240, 9999])
        left_layout.addWidget(inner_split, 1)
        # Toolbar sits just below the Display header (index 0), above the plot.
        left_layout.insertWidget(1, plot_ops_bar)

        return PlotContext(
            btn_export_plot=btn_export_plot,
            btn_assembly_tree=btn_assembly_tree,
            btn_dataset_ops=btn_dataset_ops,
            btn_settings=btn_settings,
            settings_frame=settings_frame,
            assembly_tree_panel=assembly_tree_panel,
            spin_plot_xmin=spin_plot_xmin,
            spin_plot_xmax=spin_plot_xmax,
            spin_plot_xstep=spin_plot_xstep,
            spin_plot_ymin=spin_plot_ymin,
            spin_plot_ymax=spin_plot_ymax,
            spin_plot_ystep=spin_plot_ystep,
            spin_plot_zmin=spin_plot_zmin,
            spin_plot_zmax=spin_plot_zmax,
            spin_plot_zstep=spin_plot_zstep,
            combo_plot_scale=combo_plot_scale,
            combo_polar_zero=combo_polar_zero,
            combo_colormap=combo_colormap,
            chk_colorbar=chk_colorbar,
            chk_colorbar_shared=chk_colorbar_shared,
            chk_plot_grid_visible=chk_plot_grid_visible,
            chk_colormap_invert=chk_colormap_invert,
            combo_isar_window=combo_isar_window,
            combo_isar_units=combo_isar_units,
            chk_isar_az_interp=chk_isar_az_interp,
            spin_isar_az_min=spin_isar_az_min,
            spin_isar_az_max=spin_isar_az_max,
            spin_isar_az_step=spin_isar_az_step,
            chk_isar_freq_band=chk_isar_freq_band,
            spin_isar_freq_min=spin_isar_freq_min,
            spin_isar_freq_max=spin_isar_freq_max,
            combo_isar_recon=combo_isar_recon,
            spin_isar_l1_strength=spin_isar_l1_strength,
            spin_isar_l1_iters=spin_isar_l1_iters,
            chk_isar_flip_x=chk_isar_flip_x,
            chk_isar_flip_y=chk_isar_flip_y,
            chk_isar_aperture=chk_isar_aperture,
            spin_isar_ap_center=spin_isar_ap_center,
            spin_isar_ap_width=spin_isar_ap_width,
            btn_isar_ap_prev=btn_isar_ap_prev,
            btn_isar_ap_next=btn_isar_ap_next,
            btn_isar_ap_play=btn_isar_ap_play,
            btn_isar_peak_scale=btn_isar_peak_scale,
            spin_isar_peak_drop=spin_isar_peak_drop,
            chk_isar_square=chk_isar_square,
            btn_isar_apply=btn_isar_apply,
            btn_plot_bg=btn_plot_bg,
            btn_plot_grid=btn_plot_grid,
            btn_plot_text=btn_plot_text,
            chk_plot_legend=chk_plot_legend,
            hover_readout=hover_readout,
            plot_figure=plot_figure,
            plot_canvas=plot_canvas,
            plot_ax=plot_ax,
            plot_colorbars=[],
            plot_axes=None,
            plot_bg_color=None,
            plot_grid_color=None,
            plot_text_color=None,
            last_plot_mode=None,
        )

    def _move_shared_right_panel(self, tab_key: str) -> None:
        splitter = self._plot_splitters.get(tab_key)
        if splitter is None:
            return
        if splitter.indexOf(self._shared_right_panel) >= 0:
            return
        self._shared_right_panel.setParent(None)
        self._dataset_ops_panel.setParent(None)
        splitter.insertWidget(0, self._shared_right_panel)
        splitter.insertWidget(1, self._dataset_ops_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 1)
        total = max(splitter.width(), 1400)
        ops_w = 260 if self._dataset_ops_visible else 0
        splitter.setSizes([self._dock_width, ops_w, total - self._dock_width - ops_w])
        self._dataset_ops_panel.setVisible(self._dataset_ops_visible)

    def _toggle_dataset_ops(self, checked: bool) -> None:
        """Show/hide the shared Dataset Operations panel (toggled per tab)."""
        self._dataset_ops_visible = checked
        self._dataset_ops_panel.setVisible(checked)
        splitter = self._plot_splitters.get(self._active_plot_tab)
        if splitter is None or splitter.indexOf(self._dataset_ops_panel) < 0:
            return
        sizes = splitter.sizes()
        if checked and len(sizes) >= 3 and sizes[1] == 0:
            total = sum(sizes)
            sizes[1] = 260
            sizes[2] = max(200, total - sizes[0] - 260)
            splitter.setSizes(sizes)

    def _activate_plot_tab(self, tab_key: str) -> None:
        if tab_key not in self._plot_contexts:
            return
        previous = self._plot_contexts.get(self._active_plot_tab)
        if previous is not None:
            for field in PlotContext.__dataclass_fields__:
                if hasattr(self, field):
                    setattr(previous, field, getattr(self, field))

        self._active_plot_tab = tab_key
        self._move_shared_right_panel(tab_key)

        controls = self._plot_controls_by_tab[tab_key]
        self.btn_hold = controls.get("hold")
        self.btn_clear = controls.get("clear")
        self.btn_azimuth_rect = controls.get("azimuth_rect")
        self.btn_azimuth_polar = controls.get("azimuth_polar")
        self.btn_frequency = controls.get("frequency")
        self.btn_waterfall = controls.get("waterfall")
        self.btn_fit_x = controls.get("fit_x")
        self.btn_fit_y = controls.get("fit_y")
        self.btn_auto_plot = controls.get("auto_plot")
        self.btn_auto_scale = controls.get("auto_scale")
        self.btn_pbp = controls.get("pbp")
        self.btn_isar_image = controls.get("isar_image")
        self.btn_phase = controls.get("phase")
        self.btn_zoom_box = controls.get("zoom_box")
        self.btn_pan = controls.get("pan")

        context = self._plot_contexts[tab_key]
        for field in PlotContext.__dataclass_fields__:
            setattr(self, field, getattr(context, field))

        # Keep the shared Dataset Operations panel + this tab's toggle in sync.
        self._dataset_ops_panel.setVisible(self._dataset_ops_visible)
        ops_btn = context.btn_dataset_ops
        ops_btn.blockSignals(True)
        ops_btn.setChecked(self._dataset_ops_visible)
        ops_btn.blockSignals(False)

    def _on_main_tab_changed(self, index: int) -> None:
        tab_key = self._tab_key_for_index.get(index)
        if tab_key is None:
            return
        self._activate_plot_tab(tab_key)
        self._update_plot_color_buttons()
        self.plot_canvas.draw_idle()

    def _connect_param_list(self, widget: QListWidget, axis_name: str) -> None:
        widget.itemChanged.connect(
            lambda item, axis=axis_name, lw=widget: self._on_param_item_changed(item, axis, lw)
        )

    def _on_assembly_branch_dropped(self, branch_name: str, leaf_data: list) -> None:
        """Build the dragged subtree honouring per-node add modes.

        We reach into the source AssemblyTree to recover the originating
        QTreeWidgetItem (the flat `leaf_data` list doesn't carry the
        coherent / incoherent structure). Falls back to the legacy flat
        coherent sum if the item isn't available — e.g. drag from a tree
        we can't introspect.
        """
        from assembly_tree import build_assembly_grid

        branch_item = None
        for context in self._plot_contexts.values():
            tree = getattr(context.assembly_tree_panel, "tree", None)
            if tree is None:
                continue
            candidate = getattr(tree, "_branch_drag_item", None)
            if candidate is not None and candidate.text(0) == branch_name:
                branch_item = candidate
                break

        if branch_item is not None:
            try:
                grid, history = build_assembly_grid(branch_item, axis_mode="intersect")
            except (ValueError, TypeError) as exc:
                self.status.showMessage(f"Assembly build failed: {exc}")
                return
            if grid is None:
                self.status.showMessage(
                    "Assembly branch: no loaded leaves in this subtree."
                )
                return
            self._add_dataset_row(grid, branch_name, history, file_name="")
            self.status.showMessage(f"Assembly built: {branch_name}")
            return

        # Legacy fallback: flat coherent sum of every dropped leaf.
        datasets = [(name, grid) for name, grid in leaf_data if isinstance(grid, RcsGrid)]
        skipped = len(leaf_data) - len(datasets)
        skip_msg = f" ({skipped} empty leaf(s) skipped)" if skipped else ""
        if not datasets:
            self.status.showMessage(
                "Assembly branch: no dataset data is stored in these leaves yet."
            )
            return
        if len(datasets) == 1:
            _, grid = datasets[0]
            self._add_dataset_row(grid, branch_name, f"Assembly (single): {branch_name}", file_name="")
            self.status.showMessage(f"Assembly: added {branch_name}{skip_msg}")
            return
        name_list = [n for n, _ in datasets]
        base = datasets[0][1]
        try:
            result = base.coherent_add_many(*[g for _, g in datasets[1:]])
        except (ValueError, TypeError) as exc:
            self.status.showMessage(f"Assembly coherent sum failed: {exc}")
            return
        history = "Assembly Coherent +: " + ", ".join(name_list)
        self._add_dataset_row(result, branch_name, history, file_name="")
        self.status.showMessage(f"Assembly coherent sum created: {branch_name}{skip_msg}")

    def _on_platform_built(self, name: str, grid, history: str) -> None:
        """Add a built-platform dataset to the table (signal from BuildDialog)."""
        if not isinstance(grid, RcsGrid):
            self.status.showMessage("Build platform: invalid grid returned.")
            return
        self._add_dataset_row(grid, name, history or f"Σ {name}", file_name="")
        self.status.showMessage(f"Built platform: {name}")


def main() -> int:
    app = QApplication(sys.argv)
    splash = None
    splash_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GRIM.png")
    if os.path.exists(splash_path):
        splash_pixmap = QPixmap(splash_path)
        if not splash_pixmap.isNull():
            splash = QSplashScreen(splash_pixmap, Qt.WindowStaysOnTopHint)
            splash.show()
            app.processEvents()

    window = GrimCutWindow()
    window.show()
    if splash is not None:
        QTimer.singleShot(SPLASH_DURATION_MS, lambda: splash.finish(window))
    return app.exec()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
