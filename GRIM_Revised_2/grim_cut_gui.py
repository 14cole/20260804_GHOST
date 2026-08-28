from __future__ import annotations

import base64
from collections.abc import Mapping
import os
from pathlib import Path
import sys
import tempfile
import uuid

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QByteArray, QMimeData, QSettings, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QDrag,
    QKeySequence,
    QPixmap,
    QShortcut,
)
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
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
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

from assembly_tree import MIME_BRANCH, MIME_DATASET
from assembly_workspace import AssemblyWorkspace
from feature_assembly_panel import FeatureAssemblyPanel
from freddy_integration import FreddyIntegrationWidget
from ghost_integration import GhostIntegrationWidget, load_ghost_module
from grim_dataset import RcsGrid
from grim_headless import is_supported_path
from grim_python import PythonScriptRecorder
from grim_cut_dataset_mixin import (
    DATASET_DIRTY_ROLE,
    DATASET_ID_ROLE,
    DATASET_PATH_ROLE,
    DatasetOpsMixin,
)
from grim_cut_plot_mixin import PlotOpsMixin
from ppt_workspace import DatasetCatalogEntry, PptWorkspace
from plot_models import PlotContext
from runs_workspace import RunsWorkspace

APPLICATION_PALETTES: dict[str, dict[str, object]] = {
    "Colorful": {
        "is_dark": True,
        "win_bg": "#111827",
        "panel_bg": "#19162f",
        "text": "#f8fafc",
        "head_bg": "#312e81",
        "border": "#6d28d9",
        "hover": "#0e7490",
        "checked_bg": "#7c3aed",
        "checked_border": "#22d3ee",
        "grid": "#475569",
        "muted": "#c4b5fd",
        "fg": "#f8fafc",
        "plot_line_freq": "#22d3ee",
        "plot_line_angle": "#f472b6",
        "plot_worst": "#fbbf24",
        "layer_colors": (
            "#7c3aed", "#0891b2", "#db2777", "#d97706",
            "#4f46e5", "#0e7490", "#9333ea", "#0284c7",
        ),
    },
    "Light": {
        "is_dark": False,
        "win_bg": "#f1f5f9",
        "panel_bg": "#ffffff",
        "text": "#0f172a",
        "head_bg": "#dbeafe",
        "border": "#94a3b8",
        "hover": "#bfdbfe",
        "checked_bg": "#2563eb",
        "checked_border": "#1d4ed8",
        "grid": "#cbd5e1",
        "muted": "#475569",
        "fg": "#0f172a",
        "plot_line_freq": "#0369a1",
        "plot_line_angle": "#6d28d9",
        "plot_worst": "#b45309",
        "layer_colors": (
            "#1d4ed8", "#0369a1", "#4f46e5", "#0e7490",
            "#2563eb", "#475569", "#7c3aed", "#0284c7",
        ),
    },
    "Dark": {
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
        "muted": "#94a3b8",
        "fg": "#dbeafe",
        "plot_line_freq": "#38bdf8",
        "plot_line_angle": "#a78bfa",
        "plot_worst": "#fbbf24",
        "layer_colors": (
            "#1e3a8a", "#1d4ed8", "#172554", "#2563eb",
            "#1e40af", "#3b82f6", "#334155", "#0284c7",
        ),
    },
    "Raytheon": {
        "is_dark": False,
        "win_bg": "#d9d9d6",
        "panel_bg": "#ffffff",
        "text": "#000000",
        "head_bg": "#d9d9d6",
        "border": "#63666a",
        "hover": "#63666a",
        "checked_bg": "#ce1126",
        "checked_border": "#ce1126",
        "grid": "#b1b3b3",
        "muted": "#63666a",
        "fg": "#000000",
        # Embedded application plots use primary/secondary colors. The brand's
        # tertiary colors are reserved for PowerPoint charts when needed.
        "plot_line_freq": "#000000",
        "plot_line_angle": "#ce1126",
        "plot_worst": "#63666a",
        "layer_colors": (
            "#000000", "#ce1126", "#63666a", "#b1b3b3",
            "#d9d9d6", "#000000", "#ce1126", "#63666a",
        ),
    },
}
DEFAULT_APPLICATION_PALETTE = "Dark"
APPLICATION_PALETTE_SETTINGS_KEY = "appearance/application_palette"
LEGACY_APPLICATION_PALETTE_NAMES = {
    "Raytheon-inspired": "Raytheon",
}
# Official tertiary colors are intentionally not part of application chrome
# or embedded-tool plots. They are reserved for optional PowerPoint charts
# with enough series to require additional differentiation.
RAYTHEON_TERTIARY_PPT_CHART_COLORS = (
    "#7ba7bc",
    "#b7a99a",
    "#908cc2",
    "#9abeaa",
    "#efb661",
)
# Compatibility export for extensions/tests that used GRIM's former one fixed
# palette. It remains the exact default Dark palette.
BLUE_PALETTE = APPLICATION_PALETTES[DEFAULT_APPLICATION_PALETTE]
SPLASH_DURATION_MS = 4000


def normalize_application_palette_name(value: object) -> str:
    """Return a current palette name, including legacy-setting migration."""

    normalized = str(value).strip()
    normalized = LEGACY_APPLICATION_PALETTE_NAMES.get(
        normalized,
        normalized,
    )
    if normalized not in APPLICATION_PALETTES:
        return DEFAULT_APPLICATION_PALETTE
    return normalized

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


def build_qss(palette: Mapping[str, object]) -> str:
    arrow_right = _branch_arrow_uri(
        "2,1 6,4 2,7", str(palette["text"])
    )
    arrow_down = _branch_arrow_uri(
        "1,2 7,2 4,6", str(palette["text"])
    )
    return f"""
    QMainWindow {{ background: {palette['win_bg']}; }}
    QMenuBar {{ background: {palette['panel_bg']}; color: {palette['text']}; }}
    QMenuBar::item {{ background: transparent; padding: 5px 10px; }}
    QMenuBar::item:selected {{ background: {palette['hover']}; }}
    QMenu {{
        background: {palette['panel_bg']}; color: {palette['text']};
        border: 1px solid {palette['border']};
    }}
    QMenu::item {{ padding: 6px 28px 6px 24px; }}
    QMenu::item:selected {{ background: {palette['checked_bg']}; color: white; }}
    QMenu::indicator:checked {{ background: {palette['checked_border']}; }}
    QStatusBar {{ background: {palette['panel_bg']}; color: {palette['muted']}; }}
    QWidget {{ color: {palette['text']}; }}
    QDialog {{ background: {palette['win_bg']}; color: {palette['text']}; }}
    QFrame {{ background: {palette['panel_bg']}; border: 1px solid {palette['border']}; border-radius: 8px; }}
    QFrame#paramSeparator {{
        background: {palette['border']}; min-width: 2px; max-width: 2px; border: none; border-radius: 0px;
    }}
    QGroupBox {{ color: {palette['text']}; border: 1px solid {palette['border']}; border-radius: 8px; margin-top: 10px; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
    QLabel {{ color: {palette['text']}; }}
    QTableWidget {{
        background: {palette['panel_bg']}; color: {palette['text']};
        alternate-background-color: {palette['head_bg']};
        border: 1px solid {palette['border']}; gridline-color: {palette['grid']};
    }}
    QPlainTextEdit, QTextEdit {{
        background: {palette['panel_bg']}; color: {palette['text']};
        border: 1px solid {palette['border']}; border-radius: 6px;
        selection-background-color: {palette['checked_bg']};
        selection-color: white;
    }}
    QHeaderView::section {{ background: {palette['head_bg']}; color: {palette['text']}; border: none; padding: 6px; }}
    QTabWidget::pane {{ border: 1px solid {palette['border']}; background: {palette['panel_bg']}; }}
    QTabBar::tab {{ background: {palette['panel_bg']}; color: {palette['text']}; border: 1px solid {palette['border']}; border-bottom: 0; padding: 6px 12px; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px; }}
    QTabBar::tab:selected {{ background: {palette['head_bg']}; color: {palette['text']}; border-color: {palette['checked_border']}; }}
    QTabBar::tab:hover {{ background: {palette['hover']}; }}
    QListWidget {{
        background: {palette['panel_bg']}; color: {palette['text']};
        alternate-background-color: {palette['head_bg']};
        border: 1px solid {palette['border']};
    }}
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
    QToolButton, QPushButton, QDoubleSpinBox, QSpinBox, QCheckBox,
    QRadioButton, QLineEdit, QComboBox {{
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
    QToolButton:hover, QPushButton:hover {{ border-color: {palette['hover']}; }}
    QToolButton:disabled, QPushButton:disabled, QLineEdit:disabled,
    QComboBox:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled {{
        color: {palette['muted']}; border-color: {palette['grid']};
        background: {palette['head_bg']};
    }}
    QToolButton:checked {{ background: {palette['checked_bg']}; color: white; border-color: {palette['checked_border']}; }}
    QComboBox QAbstractItemView {{ background: {palette['panel_bg']}; color: {palette['text']}; border: 1px solid {palette['border']}; }}
    QTableWidget::item:selected {{ background: {palette['checked_bg']}; color: white; }}
    QProgressBar {{
        background: {palette['head_bg']}; color: {palette['text']};
        border: 1px solid {palette['border']}; border-radius: 5px;
        text-align: center;
    }}
    QProgressBar::chunk {{ background: {palette['checked_bg']}; }}
    QSlider::groove:horizontal {{
        height: 6px; background: {palette['grid']}; border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        width: 14px; margin: -5px 0; background: {palette['checked_border']};
        border: 1px solid {palette['border']}; border-radius: 7px;
    }}
    QLabel#hoverReadout {{
        background: {palette['head_bg']}; color: {palette['text']}; border: 1px solid {palette['border']};
        border-radius: 4px; padding: 2px 6px; font-family: "Consolas","Courier New",monospace; font-size: 11px;
    }}
    QScrollArea#controlDock {{ background: {palette['win_bg']}; border: none; }}
    QScrollArea#featureAssemblyScroll {{ background: {palette['panel_bg']}; border: none; }}
    QScrollArea#plotSettingsScroll {{ background: {palette['panel_bg']}; border: none; }}
    QScrollArea#runsControlsScroll, QScrollArea#pptControlsScroll {{
        background: {palette['panel_bg']}; border: none;
    }}
    QScrollArea#runsControlsScroll > QWidget,
    QScrollArea#pptControlsScroll > QWidget,
    QWidget#runsControlsContent,
    QWidget#pptControlsContent {{ background: {palette['panel_bg']}; }}
    QWidget#featureAssemblyContent {{ background: {palette['panel_bg']}; }}
    QLabel#featurePanelIntro {{ font-size: 13px; font-weight: 600; padding: 2px 1px; }}
    QLabel#featureWorkflowSteps {{
        background: {palette['head_bg']}; color: {palette['text']};
        border: 1px solid {palette['border']}; border-radius: 6px;
        padding: 7px 9px; font-weight: 600;
    }}
    QLabel#featureNextStep {{ color: {palette['text']}; padding: 1px 4px 3px 4px; }}
    QGroupBox#featureStepCard {{
        border-color: {palette['border']}; background: {palette['panel_bg']};
        font-weight: 600;
    }}
    QGroupBox#featureStepCard QLabel, QGroupBox#featureStepCard QLineEdit,
    QGroupBox#featureStepCard QComboBox, QGroupBox#featureStepCard QCheckBox,
    QGroupBox#featureStepCard QPushButton, QGroupBox#featureStepCard QTableWidget {{
        font-weight: 400;
    }}
    QLabel#featureHint, QLabel#featureCsvSummary {{ color: {palette['muted']}; padding: 1px 2px; }}
    QLabel#featureSummary, QLabel#featureBuildSummary {{
        background: {palette['head_bg']}; border: 1px solid {palette['border']};
        border-radius: 5px; padding: 5px 7px;
    }}
    QLabel#featureContract {{
        background: {palette['head_bg']}; border-left: 3px solid {palette['checked_border']};
        border-radius: 4px; padding: 6px 8px;
    }}
    QLabel#featureReadiness {{
        font-family: "Consolas","Courier New",monospace; padding: 4px 1px;
    }}
    QLabel#featureAssemblyStatus {{
        background: {palette['head_bg']}; border: 1px solid {palette['border']};
        border-radius: 6px; padding: 6px 8px;
    }}
    QWidget#plotSettingsContent {{ background: {palette['panel_bg']}; }}
    QLabel#settingsNoMatches {{ color: {palette['muted']}; padding: 4px 2px; }}
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
        if is_supported_path(path):
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
    """Scrollable, searchable top-level popup for one plot context.

    Closing the window via its title bar untoggles the bound button so the
    button state mirrors visibility.  ``content_widget`` deliberately remains
    a plain widget: callers can keep using their existing grid layout while
    the popup owns the search bar and scroll-area plumbing.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        title: str = "Plot Settings",
    ) -> None:
        super().__init__(parent)
        self._toggle_button: QToolButton | None = None
        self._filter_rows: list[tuple[str, tuple[QWidget, ...], QWidget | None]] = []
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle(title)
        # Keep the popup useful on a compact display.  Larger contents scroll
        # in either direction instead of forcing the window beyond the screen.
        self.setMinimumSize(420, 280)
        self.resize(820, 540)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        filter_row = QHBoxLayout()
        filter_label = QLabel("Find setting")
        self.filter_edit = QLineEdit(self)
        self.filter_edit.setObjectName("plotSettingsFilter")
        self.filter_edit.setPlaceholderText("Search settings…")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.setToolTip(
            "Filter settings by label, control name, option, or help text."
        )
        filter_row.addWidget(filter_label)
        filter_row.addWidget(self.filter_edit, 1)
        outer.addLayout(filter_row)

        self.no_matches_label = QLabel("No settings match this search.")
        self.no_matches_label.setObjectName("settingsNoMatches")
        self.no_matches_label.setVisible(False)
        outer.addWidget(self.no_matches_label)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setObjectName("plotSettingsScroll")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.content_widget = QWidget()
        self.content_widget.setObjectName("plotSettingsContent")
        self.scroll_area.setWidget(self.content_widget)
        outer.addWidget(self.scroll_area, 1)

        self.filter_edit.textChanged.connect(self._apply_filter)

    @staticmethod
    def _searchable_widget_text(widget: QWidget) -> str:
        parts: list[str] = []
        for accessor in ("text", "toolTip", "placeholderText", "windowTitle"):
            method = getattr(widget, accessor, None)
            if callable(method):
                value = method()
                if value:
                    parts.append(str(value))
        if isinstance(widget, QComboBox):
            parts.extend(widget.itemText(index) for index in range(widget.count()))
        return " ".join(parts)

    def register_filter_grid(
        self,
        grid: QGridLayout,
        *,
        search_prefix: str = "",
        section: QWidget | None = None,
        excluded_widgets: tuple[QWidget, ...] = (),
    ) -> None:
        """Register each grid row as one searchable unit.

        A matching row keeps all of its labels and controls together.  When a
        nested ``section`` is supplied, the section itself disappears if none
        of its rows match, avoiding an empty ISAR block during filtering.
        """

        excluded_ids = {id(widget) for widget in excluded_widgets}
        for row in range(grid.rowCount()):
            widgets: list[QWidget] = []
            seen: set[int] = set()
            for column in range(grid.columnCount()):
                item = grid.itemAtPosition(row, column)
                widget = item.widget() if item is not None else None
                if widget is None or id(widget) in excluded_ids or id(widget) in seen:
                    continue
                seen.add(id(widget))
                widgets.append(widget)
            if not widgets:
                continue
            haystack = " ".join(
                [search_prefix]
                + [self._searchable_widget_text(widget) for widget in widgets]
            ).casefold()
            self._filter_rows.append((haystack, tuple(widgets), section))

    def _apply_filter(self, text: str) -> None:
        tokens = tuple(part.casefold() for part in text.split() if part)
        section_matches: dict[QWidget, bool] = {}
        any_match = False
        for haystack, widgets, section in self._filter_rows:
            matches = all(token in haystack for token in tokens)
            any_match = any_match or matches
            for widget in widgets:
                widget.setVisible(matches)
            if section is not None:
                section_matches[section] = section_matches.get(section, False) or matches
        for section, matches in section_matches.items():
            section.setVisible(matches)
        self.no_matches_label.setVisible(bool(tokens) and not any_match)

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
    def _build_application_palette_menu(self) -> None:
        """Expose one discoverable, exclusive application-theme selector."""

        view_menu = self.menuBar().addMenu("View")
        self.application_palette_menu = view_menu.addMenu(
            "Application Palette"
        )
        self.application_palette_menu.setObjectName(
            "applicationPaletteMenu"
        )
        self.application_palette_group = QActionGroup(self)
        self.application_palette_group.setExclusive(True)
        descriptions = {
            "Colorful": "Purple, cyan, and magenta dark application chrome",
            "Light": "Bright neutral application chrome with blue accents",
            "Dark": "GRIM blue/slate dark application chrome",
            "Raytheon": "Official white, black, cool gray, and Red 186 chrome",
        }
        for palette_name in APPLICATION_PALETTES:
            action = QAction(palette_name, self)
            action.setCheckable(True)
            action.setData(palette_name)
            action.setStatusTip(descriptions[palette_name])
            action.setChecked(
                palette_name == self.application_palette_name
            )
            action.triggered.connect(
                lambda checked=False, name=palette_name: (
                    self._apply_application_palette(name)
                    if checked
                    else None
                )
            )
            self.application_palette_group.addAction(action)
            self.application_palette_menu.addAction(action)
            self._application_palette_actions[palette_name] = action

    def _apply_application_palette(
        self,
        palette_name: str,
        *,
        persist: bool = True,
    ) -> None:
        """Apply one palette to GRIM and every embedded workspace."""

        normalized = normalize_application_palette_name(palette_name)
        self.application_palette_name = normalized
        self.application_palette = dict(APPLICATION_PALETTES[normalized])

        for name, action in self._application_palette_actions.items():
            previous = action.blockSignals(True)
            action.setChecked(name == normalized)
            action.blockSignals(previous)

        self.setStyleSheet(build_qss(self.application_palette))

        ghost = getattr(self, "ghost_integration", None)
        apply_ghost_palette = getattr(
            ghost, "apply_application_palette", None
        )
        if callable(apply_ghost_palette):
            apply_ghost_palette(self.application_palette)

        assembly_workspace = getattr(self, "assembly_workspace", None)
        apply_assembly_palette = getattr(
            assembly_workspace, "apply_application_palette", None
        )
        if callable(apply_assembly_palette):
            apply_assembly_palette(self.application_palette)

        freddy = getattr(self, "freddy_integration", None)
        apply_freddy_palette = getattr(
            freddy, "apply_application_palette", None
        )
        if callable(apply_freddy_palette):
            apply_freddy_palette(self.application_palette)

        # Plot background/grid/text follow the application palette unless the
        # user explicitly chose an override with the three Plot Colors buttons.
        for context in getattr(self, "_plot_contexts", {}).values():
            self._apply_palette_to_plot_context(context)

        if persist:
            self._settings.setValue(
                APPLICATION_PALETTE_SETTINGS_KEY,
                normalized,
            )
            sync = getattr(self._settings, "sync", None)
            if callable(sync):
                sync()
            status = getattr(self, "status", None)
            if status is not None:
                status.showMessage(f"Application palette: {normalized}")

    def _apply_palette_to_plot_context(self, context: PlotContext) -> None:
        """Restyle one plot context without changing tabs or popup state."""

        background = str(
            context.plot_bg_color or self.application_palette["panel_bg"]
        )
        grid = str(
            context.plot_grid_color or self.application_palette["grid"]
        )
        text = str(
            context.plot_text_color or self.application_palette["text"]
        )
        context.plot_figure.set_facecolor(background)
        for figure_text in context.plot_figure.texts:
            figure_text.set_color(text)
        figure_title = getattr(context.plot_figure, "_suptitle", None)
        if figure_title is not None:
            figure_title.set_color(text)
        axes = context.plot_axes or [context.plot_ax]
        for axis in axes:
            axis.set_facecolor(background)
            if context.chk_plot_grid_visible.isChecked():
                axis.grid(True, color=grid, alpha=0.35)
            else:
                axis.grid(False)
            axis.tick_params(colors=text)
            axis.title.set_color(text)
            axis.xaxis.label.set_color(text)
            axis.yaxis.label.set_color(text)
            if hasattr(axis, "zaxis") and axis.zaxis is not None:
                axis.zaxis.label.set_color(text)
            for spine in getattr(axis, "spines", {}).values():
                spine.set_color(str(self.application_palette["border"]))
            for annotation in axis.texts:
                annotation.set_color(text)
            legend = axis.get_legend()
            if legend is not None:
                for label in legend.get_texts():
                    label.set_color(text)
                legend.get_title().set_color(text)
                legend.get_frame().set_facecolor(background)
                legend.get_frame().set_edgecolor(grid)
        for colorbar in context.plot_colorbars:
            colorbar.ax.set_facecolor(background)
            colorbar.ax.tick_params(colors=text)
            colorbar.ax.yaxis.label.set_color(text)
            colorbar.ax.xaxis.label.set_color(text)
            colorbar.outline.set_edgecolor(grid)
        context.btn_plot_bg.setStyleSheet(f"background: {background};")
        context.btn_plot_grid.setStyleSheet(f"background: {grid};")
        context.btn_plot_text.setStyleSheet(f"background: {text};")
        context.plot_canvas.draw_idle()

    def __init__(self, *, settings: QSettings | None = None) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self._settings = settings if settings is not None else QSettings(
            "GRIM", "GRIM"
        )
        saved_palette_value = str(
            self._settings.value(
                APPLICATION_PALETTE_SETTINGS_KEY,
                DEFAULT_APPLICATION_PALETTE,
            )
        )
        saved_palette = normalize_application_palette_name(
            saved_palette_value
        )
        if saved_palette_value in LEGACY_APPLICATION_PALETTE_NAMES:
            self._settings.setValue(
                APPLICATION_PALETTE_SETTINGS_KEY,
                saved_palette,
            )
            sync = getattr(self._settings, "sync", None)
            if callable(sync):
                sync()
        self.application_palette_name = saved_palette
        self.application_palette = dict(APPLICATION_PALETTES[saved_palette])
        self._application_palette_actions: dict[str, QAction] = {}

        self.setWindowTitle("GRIM Cut")
        self.resize(1550, 900)
        self._dock_width = 480
        self._build_application_palette_menu()

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
        self.btn_dataset_load = QToolButton(text="Load…")
        self.btn_dataset_save = QToolButton(text="Save")
        self.btn_dataset_save_all = QToolButton(text="Save All")
        self.btn_dataset_delete = QToolButton(text="Delete")
        dataset_actions.addWidget(self.btn_dataset_load)
        dataset_actions.addWidget(self.btn_dataset_save)
        dataset_actions.addWidget(self.btn_dataset_save_all)
        dataset_actions.addWidget(self.btn_dataset_delete)
        dataset_actions.addStretch(1)
        sec_datasets.addLayout(dataset_actions)

        self.table = DatasetTable(0, 3)
        self.table.setHorizontalHeaderLabels(["Name", "Source / Output", "History"])
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
            # Edits are committed through RcsGrid.edit_axis_value(), which
            # validates the new coordinate/label and transactionally reorders
            # every aligned sample array when a numeric axis changes order.
            widget.setEditTriggers(
                QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
            )
            widget.setToolTip(
                "Double-click a value or press F2 to edit it. Numeric axes are "
                "kept sorted and their samples move with the edited coordinate."
            )
            widget.setMinimumHeight(96)
        self.lbl_pol = ClickableLabel("Polarization")
        self.lbl_freq = ClickableLabel("Frequency")
        self.lbl_elev = ClickableLabel("Elevation")
        self.lbl_az = ClickableLabel("Azimuth")
        lbl_pol = self.lbl_pol
        lbl_freq = self.lbl_freq
        lbl_elev = self.lbl_elev
        lbl_az = self.lbl_az
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
        _ops_pad("Calibration", (
            ("Range Cal", "btn_range_cal"),
        ), cols=1)
        self.btn_overlap.setToolTip(
            "Crop every selected dataset to the common axis values and finite "
            "cells shared by all selected datasets. Selection order does not "
            "choose a reference dataset."
        )
        self.btn_range_cal.setToolTip(
            "Complex substitution calibration using loaded measured-cal and "
            "exact-reference datasets plus a signed one-way physical "
            "displacement; GRIM applies the monostatic two-way phase."
        )
        _ops_pad("Geometry & Units", (
            ("El→Az360", "btn_el_to_az360"),
            ("Swap El/Az", "btn_swap_el_az"),
            ("SENTRi El→GRIM", "btn_sentri_elevation"),
            ("→ dBke", "btn_to_dbke"),
            ("→ dBsm", "btn_to_dbsm"),
            ("Conic ↔ GC (0°)", "btn_conic_gc"),
            ("Wedge → Conic", "btn_wedge_to_conic"),
        ))
        self.btn_sentri_elevation.setToolTip(
            "Convert selected native SENTRi polar Theta to GRIM signed "
            "elevation: elevation = 90° - Theta (0° waterline, +90° "
            "top-down, -90° bottom-up). Samples are reordered with the "
            "monotonically increasing elevation axis; no interpolation or "
            "phase change is applied."
        )
        self.btn_conic_gc.setToolTip(
            "Exact 0° Conic/Great-Circle relabel only. General GC cuts are "
            "blocked because they require curved-path complex interpolation "
            "and polarization-basis rotation."
        )
        self.btn_wedge_to_conic.setToolTip(
            "Convert a vertical-turntable/body-y-wedge acquisition into the "
            "normal tilted-pylon conic grid. Requires a full revolution, at "
            "least two measured wedge tilts, finite complex phase, and VV/HH "
            "plus VH or HV unless zero cross-pol is explicitly assumed. "
            "Unsupported side-aspect/elevation combinations remain NaN."
        )

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

        # PPT owns an independent report selection and a true 16:9 slide
        # preview. It consumes the same in-memory RcsGrid objects but does not
        # reparent or alter the Plotting tab's dataset/parameter controls.
        self.ppt_workspace = PptWorkspace(
            self,
            selected_ids_provider=self._selected_dataset_ids_for_ppt,
        )
        # One canonical Assembly workspace replaces the two independent,
        # hidden trees that used to live inside the Plotting and ISAR views.
        self.assembly_workspace = AssemblyWorkspace(self)
        self.feature_assembly_panel = FeatureAssemblyPanel(
            self.assembly_workspace
        )
        self.assembly_workspace.set_feature_controls(
            self.feature_assembly_panel
        )
        assembly_tree_panel = getattr(
            self.assembly_workspace, "assembly_tree_panel", None
        )
        assembly_dirty_changed = getattr(
            assembly_tree_panel, "dirty_changed", None
        )
        if callable(getattr(assembly_dirty_changed, "connect", None)):
            assembly_dirty_changed.connect(
                lambda dirty: self._set_main_tab_dirty(
                    self.assembly_workspace, "Assembly", dirty
                )
            )

        # GHOST remains optional at runtime. The integration widget shows an
        # actionable unavailable message when its backend is not installed.
        self.ghost_integration = GhostIntegrationWidget(self)

        # FREDDY is a material/IBC design workspace, not an RCS dataset
        # producer. Keep it as an independent top-level tool tab and do not
        # connect its CSV outputs to GRIM's RCS dataset loader.
        self.freddy_integration = FreddyIntegrationWidget(self)

        # Remote HPC requests are declarative on Windows and become final,
        # provenance-bound GHOST runs only after staging on the Linux login
        # node.  The workspace also keeps a manual bundle-export path for
        # clusters whose VPN/MFA policy blocks non-interactive SSH.
        self.runs_workspace = RunsWorkspace(
            self,
            backend_path=self.ghost_integration.backend_path,
        )
        runs_controls_content = getattr(
            self.runs_workspace, "controls_content", None
        )
        if runs_controls_content is not None:
            runs_controls_content.setObjectName("runsControlsContent")

        # A deliberately small, read-only view of the semantic dataset/plot
        # operations performed in this session.  The recorder ignores UI
        # navigation, selection gestures, zooming, and solver/tool tabs.
        self.tab_python = QWidget()
        python_layout = QVBoxLayout(self.tab_python)
        python_layout.setContentsMargins(14, 14, 14, 14)
        python_layout.setSpacing(8)

        python_header = QHBoxLayout()
        python_title = QLabel("Headless dataset and plot script")
        python_title.setObjectName("plotTitle")
        python_header.addWidget(python_title)
        python_header.addStretch(1)
        self.btn_python_clear = QPushButton("Clear")
        self.btn_python_copy = QPushButton("Copy")
        self.btn_python_save = QPushButton("Save As…")
        python_header.addWidget(self.btn_python_clear)
        python_header.addWidget(self.btn_python_copy)
        python_header.addWidget(self.btn_python_save)
        python_layout.addLayout(python_header)

        python_hint = QLabel(
            "Successful dataset operations and supported rectangular/polar azimuth, "
            "frequency, and elevation-sweep plots appear here. PBP and other plot "
            "modes are noted but not emitted as runnable code. Tab changes, zoom, "
            "pan, and selection gestures are not recorded."
        )
        python_hint.setWordWrap(True)
        python_layout.addWidget(python_hint)
        self.python_script_view = QPlainTextEdit()
        self.python_script_view.setReadOnly(True)
        self.python_script_view.setLineWrapMode(
            QPlainTextEdit.LineWrapMode.NoWrap
        )
        python_layout.addWidget(self.python_script_view, 1)

        self._python_clean_script: str | None = None
        self._python_empty_script = ""
        self.python_recorder = PythonScriptRecorder(
            self._on_python_script_changed
        )
        self._python_empty_script = self.python_recorder.script
        self._python_clean_script = self.python_recorder.script

        # Keep registration centralized so construction dependencies do not
        # dictate the user-facing workflow order.
        for label, widget, plot_key in (
            ("Plotting", self.tab_simple_plots, "plotting"),
            ("ISAR", self.tab_isar, "isar"),
            ("FREDDY", self.freddy_integration, None),
            ("GHOST", self.ghost_integration, None),
            ("Assembly", self.assembly_workspace, None),
            ("PPT", self.ppt_workspace, None),
            ("Runs", self.runs_workspace, None),
            ("Python", self.tab_python, None),
        ):
            tab_index = self.main_tabs.addTab(widget, label)
            if plot_key is not None:
                self._tab_key_for_index[tab_index] = plot_key
        self._sync_python_tab_title()

        try:
            feature_service = load_ghost_module(
                "feature_workflow", self.ghost_integration.backend_path
            )
            self.feature_assembly_panel.set_service(feature_service)
        except Exception as exc:
            self.assembly_workspace.lbl_status.setText(
                "Feature assembly backend is unavailable: " + str(exc)
            )

        self.status = self.statusBar()
        self.status.showMessage("Ready")

        self.btn_python_clear.clicked.connect(self._clear_python_script)
        self.btn_python_copy.clicked.connect(self._copy_python_script)
        self.btn_python_save.clicked.connect(self._save_python_script)

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

        self.table.files_dropped.connect(self._handle_files_dropped)
        self.table.assembly_branch_dropped.connect(self._on_assembly_branch_dropped)
        self.table.rows_reordered.connect(self._on_dataset_rows_reordered)
        self.assembly_workspace.files_to_load.connect(self._handle_files_dropped)
        self.assembly_workspace.platform_built.connect(self._on_platform_built)
        self.assembly_workspace.feature_built.connect(
            self._on_assembly_feature_built
        )
        self.assembly_workspace.feature_build_failed.connect(
            self.status.showMessage
        )
        self.feature_assembly_panel.preview_ready.connect(
            self._on_feature_preview_ready
        )
        self.feature_assembly_panel.preview_stale.connect(
            self.assembly_workspace.mark_preview_stale
        )
        self.feature_assembly_panel.feature_built.connect(
            self._on_feature_file_built
        )
        self.feature_assembly_panel.build_failed.connect(
            self.status.showMessage
        )
        self.feature_assembly_panel.status_changed.connect(
            self.status.showMessage
        )
        self.ppt_workspace.status_changed.connect(self.status.showMessage)
        self.ppt_workspace.report_exported.connect(
            lambda path: self.status.showMessage(
                f"PowerPoint report saved: {path}"
            )
        )
        self.ghost_integration.files_exported.connect(
            self._on_ghost_files_exported
        )
        self.runs_workspace.status_changed.connect(self.status.showMessage)
        self.runs_workspace.results_downloaded.connect(
            self._on_hpc_results_downloaded
        )
        freddy_attach = getattr(
            self.freddy_integration, "attach_to_ghost_requested", None
        )
        ghost_attach = getattr(
            self.ghost_integration, "attach_material_artifact", None
        )
        if callable(getattr(freddy_attach, "connect", None)) and callable(
            ghost_attach
        ):
            freddy_attach.connect(ghost_attach)
        self.table.itemSelectionChanged.connect(self._on_dataset_selection_changed)
        self.table.itemChanged.connect(self._on_dataset_table_item_changed)
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
            for recorded_mode in (
                "azimuth_rect",
                "azimuth_polar",
                "frequency",
                "elevation_sweep",
                "waterfall",
                "compare",
                "isar_image",
                "az_vs_range",
            ):
                button = controls.get(recorded_mode)
                if button is not None:
                    button.clicked.connect(
                        lambda _checked=False, mode=recorded_mode: (
                            self._on_explicit_plot_clicked(mode)
                        )
                    )

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
        self.btn_range_cal.clicked.connect(self._range_cal_selected)
        self.btn_medianize.clicked.connect(self._medianize_selected)
        self.btn_duplicate.clicked.connect(self._duplicate_selected)
        self.btn_el_to_az360.clicked.connect(self._elevation_to_azimuth_360_selected)
        self.btn_swap_el_az.clicked.connect(self._swap_elevation_azimuth_selected)
        self.btn_sentri_elevation.clicked.connect(
            self._convert_sentri_elevation_selected
        )
        self.btn_to_dbke.clicked.connect(self._convert_to_dbke_selected)
        self.btn_to_dbsm.clicked.connect(self._convert_to_dbsm_selected)
        self.btn_conic_gc.clicked.connect(self._convert_conic_gc_selected)
        self.btn_wedge_to_conic.clicked.connect(self._convert_wedge_to_conic_selected)
        self.btn_dataset_load.clicked.connect(self._load_dataset_files)
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
            # On the ISAR tab every settings-frame change re-runs the FFT
            # imaging, which is too slow for per-keystroke spinbox updates.
            # Defer everything through the Apply button instead — EXCEPT the
            # color-scale spinboxes, which only retune the existing image's
            # clim (no recompute) and so can stay live.
            # The plotting tab keeps live updates because its plots are cheap.
            # Aperture scrubbing and peak-scaling stay live inside ISAR —
            # stepping/playing is the point of the scrub workflow, and the
            # async compute path coalesces bursts of requests.
            if tab_key == "isar":
                context.btn_isar_apply.clicked.connect(self._on_isar_window_changed)
                context.btn_isar_ap_prev.clicked.connect(self._on_isar_ap_prev)
                context.btn_isar_ap_next.clicked.connect(self._on_isar_ap_next)
                context.btn_isar_ap_play.toggled.connect(self._on_isar_ap_play)
                context.btn_isar_peak_scale.clicked.connect(self._on_isar_peak_scale)
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

        self._activate_plot_tab("plotting")
        self._apply_application_palette(
            self.application_palette_name,
            persist=False,
        )
        self.main_tabs.currentChanged.connect(self._on_main_tab_changed)
        self._notify_dataset_catalog_changed()
        self._update_plot_color_buttons()

    def _on_python_script_changed(self, script: str) -> None:
        self.python_script_view.setPlainText(script)
        self._sync_python_tab_title(script)

    def _python_script_is_dirty(self) -> bool:
        clean = self._python_clean_script
        return clean is not None and self.python_recorder.script != clean

    def _sync_python_tab_title(self, script: str | None = None) -> None:
        if not hasattr(self, "main_tabs") or not hasattr(self, "tab_python"):
            return
        index = self.main_tabs.indexOf(self.tab_python)
        if index < 0:
            return
        current = self.python_recorder.script if script is None else script
        dirty = self._python_clean_script is not None and current != self._python_clean_script
        self.main_tabs.setTabText(index, "Python*" if dirty else "Python")

    def _set_main_tab_dirty(
        self, widget: QWidget, clean_title: str, dirty: bool
    ) -> None:
        index = self.main_tabs.indexOf(widget)
        if index >= 0:
            self.main_tabs.setTabText(
                index, f"{clean_title}*" if dirty else clean_title
            )

    def _clear_python_script(self) -> None:
        if self.python_recorder.script != self._python_empty_script:
            buttons = getattr(QMessageBox, "StandardButton", QMessageBox)
            answer = QMessageBox.question(
                self,
                "Clear Recorded Python Script?",
                "Clear every recorded dataset operation and plot command? "
                "This cannot be undone.",
                buttons.Yes | buttons.No,
                buttons.No,
            )
            if answer != buttons.Yes:
                return
        self.python_recorder.clear()
        # Clear is an explicit, confirmed destructive action. Treat the empty
        # recorder as the new clean state so closing does not prompt again.
        self._python_clean_script = self.python_recorder.script
        self._sync_python_tab_title()
        self.status.showMessage("Python script cleared.")

    def _copy_python_script(self) -> None:
        QApplication.clipboard().setText(self.python_recorder.script)
        self.status.showMessage("Python script copied to the clipboard.")

    def _save_python_script(self) -> bool:
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Python Script",
            "grim_workflow.py",
            "Python Files (*.py);;All Files (*)",
        )
        if not path:
            return False
        if not path.lower().endswith(".py"):
            path = f"{path}.py"
        target = Path(path).expanduser().resolve(strict=False)
        temporary_path: Path | None = None
        fd = -1
        try:
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                fd = -1
                stream.write(self.python_recorder.script)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, target)
            temporary_path = None
        except OSError as exc:
            if fd >= 0:
                os.close(fd)
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
            QMessageBox.critical(self, "Save Python Script Failed", str(exc))
            self.status.showMessage(f"Python script save failed: {exc}")
            return False
        self._python_clean_script = self.python_recorder.script
        self._sync_python_tab_title()
        self.status.showMessage(f"Python script saved: {target}")
        return True

    def _confirm_python_script_close(self) -> bool:
        if not self._python_script_is_dirty():
            return True
        buttons = getattr(QMessageBox, "StandardButton", QMessageBox)
        answer = QMessageBox.warning(
            self,
            "Unsaved Python Recorder Script",
            "The Python recorder contains changes that have not been saved. "
            "Save the script before closing GRIM?",
            buttons.Save | buttons.Discard | buttons.Cancel,
            buttons.Save,
        )
        if answer == buttons.Cancel:
            return False
        if answer == buttons.Save:
            return self._save_python_script()
        return True

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
        btn_dataset_ops = QToolButton(text="Dataset Operations")
        btn_dataset_ops.setCheckable(True)
        btn_export_plot = QToolButton(text="Export Plot")
        settings_title = "ISAR Settings" if tab_key == "isar" else "Plot Settings"
        btn_settings = QToolButton(text=settings_title)
        btn_settings.setCheckable(True)
        topbar.addWidget(btn_dataset_ops)
        topbar.addWidget(btn_export_plot)
        topbar.addWidget(btn_settings)
        left_layout.addLayout(topbar)

        settings_frame = PlotSettingsPopup(panel, title=settings_title)
        settings_frame.setObjectName(f"{tab_key}SettingsPopup")
        settings_frame.setFrameShape(QFrame.StyledPanel)
        settings_frame.setVisible(False)
        settings_frame.set_toggle_button(btn_settings)
        settings_layout = QGridLayout(settings_frame.content_widget)
        settings_layout.setContentsMargins(8, 8, 8, 8)
        settings_layout.setHorizontalSpacing(8)
        settings_layout.setVerticalSpacing(6)
        settings_layout.setColumnStretch(1, 1)
        settings_layout.setColumnStretch(3, 1)
        settings_layout.setColumnStretch(5, 1)

        row = 0
        plotting_only_rows: list[int] = []
        plotting_only_rows.append(row)
        settings_layout.addWidget(QLabel("Plot X Min"), row, 0)
        spin_plot_xmin = QDoubleSpinBox()
        spin_plot_xmin.setRange(-1e9, 1e9)
        spin_plot_xmin.setDecimals(6)
        spin_plot_xmin.setValue(-180.0)
        settings_layout.addWidget(spin_plot_xmin, row, 1)
        settings_layout.addWidget(QLabel("Plot X Max"), row, 2)
        spin_plot_xmax = QDoubleSpinBox()
        spin_plot_xmax.setRange(-1e9, 1e9)
        spin_plot_xmax.setDecimals(6)
        spin_plot_xmax.setValue(180.0)
        settings_layout.addWidget(spin_plot_xmax, row, 3)
        settings_layout.addWidget(QLabel("Plot X Step"), row, 4)
        spin_plot_xstep = QDoubleSpinBox()
        spin_plot_xstep.setRange(0.0, 1e9)
        spin_plot_xstep.setDecimals(6)
        spin_plot_xstep.setValue(0.0)
        settings_layout.addWidget(spin_plot_xstep, row, 5)
        row += 1

        plotting_only_rows.append(row)
        settings_layout.addWidget(QLabel("Plot Y Min"), row, 0)
        spin_plot_ymin = QDoubleSpinBox()
        spin_plot_ymin.setRange(-1e9, 1e9)
        spin_plot_ymin.setDecimals(12)
        spin_plot_ymin.setValue(-80.0)
        settings_layout.addWidget(spin_plot_ymin, row, 1)
        settings_layout.addWidget(QLabel("Plot Y Max"), row, 2)
        spin_plot_ymax = QDoubleSpinBox()
        spin_plot_ymax.setRange(-1e9, 1e9)
        spin_plot_ymax.setDecimals(12)
        spin_plot_ymax.setValue(0.0)
        settings_layout.addWidget(spin_plot_ymax, row, 3)
        settings_layout.addWidget(QLabel("Plot Y Step"), row, 4)
        spin_plot_ystep = QDoubleSpinBox()
        spin_plot_ystep.setRange(0.0, 1e9)
        spin_plot_ystep.setDecimals(6)
        spin_plot_ystep.setValue(0.0)
        settings_layout.addWidget(spin_plot_ystep, row, 5)
        row += 1

        z_label = "Color" if tab_key == "isar" else "Plot Z"
        settings_layout.addWidget(QLabel(f"{z_label} Min"), row, 0)
        spin_plot_zmin = QDoubleSpinBox()
        spin_plot_zmin.setRange(-1e9, 1e9)
        spin_plot_zmin.setDecimals(12)
        spin_plot_zmin.setValue(0.0)
        settings_layout.addWidget(spin_plot_zmin, row, 1)
        settings_layout.addWidget(QLabel(f"{z_label} Max"), row, 2)
        spin_plot_zmax = QDoubleSpinBox()
        spin_plot_zmax.setRange(-1e9, 1e9)
        spin_plot_zmax.setDecimals(12)
        spin_plot_zmax.setValue(0.0)
        settings_layout.addWidget(spin_plot_zmax, row, 3)
        settings_layout.addWidget(QLabel(f"{z_label} Step"), row, 4)
        spin_plot_zstep = QDoubleSpinBox()
        spin_plot_zstep.setRange(0.0, 1e9)
        spin_plot_zstep.setDecimals(6)
        spin_plot_zstep.setValue(0.0)
        settings_layout.addWidget(spin_plot_zstep, row, 5)
        row += 1

        scale_label = "Image Scale" if tab_key == "isar" else "Plot Scale"
        settings_layout.addWidget(QLabel(scale_label), row, 0)
        combo_plot_scale = QComboBox()
        combo_plot_scale.addItem("dBsm", "dbsm")
        combo_plot_scale.addItem("Linear", "linear")
        default_index = combo_plot_scale.findData("dbsm")
        if default_index >= 0:
            combo_plot_scale.setCurrentIndex(default_index)
        settings_layout.addWidget(combo_plot_scale, row, 1, 1, 5)
        row += 1

        plotting_only_rows.append(row)
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

        settings_layout.addWidget(QLabel("Plot Colormap"), row, 0)
        combo_colormap = QComboBox()
        for name in ("viridis", "plasma", "inferno", "magma", "cividis", "turbo"):
            combo_colormap.addItem(name, name)
        settings_layout.addWidget(combo_colormap, row, 1)
        chk_colorbar = QCheckBox("Show Colorbar")
        chk_colorbar.setChecked(True)
        settings_layout.addWidget(chk_colorbar, row, 2)
        chk_colorbar_shared = QCheckBox("Shared Colorbar")
        chk_colorbar_shared.setChecked(True)
        settings_layout.addWidget(chk_colorbar_shared, row, 3)
        row += 1

        # ISAR formation controls live in their own nested section.  The
        # section is only inserted into the ISAR popup; the PlotContext still
        # owns compatible widgets so its public API remains unchanged.
        common_settings_layout = settings_layout
        common_row = row
        isar_settings_section = QWidget(settings_frame.content_widget)
        isar_settings_section.setObjectName("isarSettingsSection")
        settings_layout = QGridLayout(isar_settings_section)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setHorizontalSpacing(8)
        settings_layout.setVerticalSpacing(6)
        settings_layout.setColumnStretch(1, 1)
        settings_layout.setColumnStretch(3, 1)
        settings_layout.setColumnStretch(5, 1)
        row = 0

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
        combo_isar_recon.addItems([
            "Fast PFA (FFT)",
            "Accurate PFA (Cartesian)",
            "Sparse L1 (clean)",
        ])
        combo_isar_recon.setToolTip(
            "Fast PFA: keystone-corrected matched-filter imaging — quickest, "
            "but residual range curvature can defocus off-centre scatterers.\n"
            "Accurate PFA: regrids both Cartesian wavenumber axes before the FFT; "
            "better focus for wider apertures and targets away from the phase centre.\n"
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

        isar_settings_layout = settings_layout
        settings_layout = common_settings_layout
        row = common_row
        if tab_key == "isar":
            settings_layout.addWidget(isar_settings_section, row, 0, 1, 6)
            row += 1
        else:
            isar_settings_section.setVisible(False)

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

        common_filter_exclusions = [isar_settings_section]
        if tab_key == "isar":
            # X/Y limits are managed by the ISAR Fit/Zoom/Pan tools and polar-
            # zero direction does not participate in ISAR formation.  Keep
            # the backing widgets for PlotContext API compatibility, but do
            # not present those inert Plotting-only rows.  Image Scale remains
            # visible because ISAR honors its dBsm/linear selection on Apply.
            for plotting_only_row in plotting_only_rows:
                seen: set[int] = set()
                for column in range(settings_layout.columnCount()):
                    item = settings_layout.itemAtPosition(plotting_only_row, column)
                    widget = item.widget() if item is not None else None
                    if widget is None or id(widget) in seen:
                        continue
                    seen.add(id(widget))
                    widget.setVisible(False)
                    common_filter_exclusions.append(widget)

        settings_frame.register_filter_grid(
            settings_layout,
            excluded_widgets=tuple(common_filter_exclusions),
        )
        if tab_key == "isar":
            settings_frame.register_filter_grid(
                isar_settings_layout,
                search_prefix="ISAR",
                section=isar_settings_section,
            )

        plot_frame = QFrame()
        plot_frame.setFrameShape(QFrame.StyledPanel)
        plot_layout = QVBoxLayout(plot_frame)
        plot_layout.setContentsMargins(20, 20, 20, 20)
        plot_layout.setSpacing(12)

        palette = getattr(self, "application_palette", BLUE_PALETTE)
        plot_figure = Figure(facecolor=palette["panel_bg"])
        plot_canvas = FigureCanvas(plot_figure)
        plot_canvas.setMinimumSize(320, 240)
        plot_canvas.setStyleSheet("background: transparent;")
        plot_ax = plot_figure.add_subplot(111)
        plot_ax.set_facecolor(palette["panel_bg"])
        plot_ax.grid(True, color=palette["grid"], alpha=0.35)
        plot_ax.tick_params(colors=palette["text"])
        plot_ax.xaxis.label.set_color(palette["text"])
        plot_ax.yaxis.label.set_color(palette["text"])
        for spine in plot_ax.spines.values():
            spine.set_color(palette["border"])
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

        left_layout.addWidget(plot_frame, 1)
        # Toolbar sits just below the Display header (index 0), above the plot.
        left_layout.insertWidget(1, plot_ops_bar)

        return PlotContext(
            btn_export_plot=btn_export_plot,
            btn_dataset_ops=btn_dataset_ops,
            btn_settings=btn_settings,
            settings_frame=settings_frame,
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
            if tab_key != self._active_plot_tab and previous.settings_frame.isVisible():
                # A settings frame is a top-level popup.  Close the old tab's
                # popup during a Plotting/ISAR switch so two independently
                # filtered settings windows never overlap or imply that one
                # tab controls the other.
                previous.settings_frame.close()

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
            active = self._plot_contexts.get(self._active_plot_tab)
            if active is not None and active.settings_frame.isVisible():
                active.settings_frame.close()
            if self.main_tabs.widget(index) is self.ppt_workspace:
                self._notify_dataset_catalog_changed()
            return
        self._activate_plot_tab(tab_key)
        self._update_plot_color_buttons()
        self.plot_canvas.draw_idle()

    def _connect_param_list(self, widget: QListWidget, axis_name: str) -> None:
        widget.itemChanged.connect(
            lambda item, axis=axis_name, axis_widget=widget: (
                self._on_param_item_changed(item, axis, axis_widget)
            )
        )

    def _dataset_catalog(self) -> tuple[DatasetCatalogEntry, ...]:
        """Return stable, ordered dataset references for report consumers."""

        entries: list[DatasetCatalogEntry] = []
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            if name_item is None:
                continue
            dataset = name_item.data(Qt.UserRole)
            if not isinstance(dataset, RcsGrid):
                continue
            dataset_id = name_item.data(DATASET_ID_ROLE)
            if not dataset_id:
                dataset_id = uuid.uuid4().hex
                name_item.setData(DATASET_ID_ROLE, dataset_id)
            file_item = self.table.item(row, 1)
            source = ""
            is_dirty = bool(name_item.data(DATASET_DIRTY_ROLE))
            if not is_dirty and file_item is not None:
                source = str(file_item.data(DATASET_PATH_ROLE) or "").strip()
            if not is_dirty and not source:
                source = str(getattr(dataset, "source_path", "") or "")
            if not is_dirty and not source and file_item is not None:
                displayed = file_item.text().strip()
                if displayed.casefold() != "unsaved":
                    source = displayed
            entries.append(
                DatasetCatalogEntry(
                    str(dataset_id),
                    name_item.text().strip() or f"Dataset {row + 1}",
                    dataset,
                    source,
                )
            )
        return tuple(entries)

    def _selected_dataset_ids_for_ppt(self) -> tuple[str, ...]:
        """Snapshot the main table selection without changing either view."""

        ids: list[str] = []
        for index in sorted(
            self.table.selectionModel().selectedRows(), key=lambda value: value.row()
        ):
            item = self.table.item(index.row(), 0)
            if item is None:
                continue
            dataset_id = item.data(DATASET_ID_ROLE)
            if dataset_id:
                ids.append(str(dataset_id))
        return tuple(ids)

    def _notify_dataset_catalog_changed(self) -> None:
        catalog = self._dataset_catalog()
        workspace = getattr(self, "ppt_workspace", None)
        if workspace is not None:
            workspace.set_dataset_catalog(catalog)
        feature_panel = getattr(self, "feature_assembly_panel", None)
        set_feature_catalog = getattr(
            feature_panel, "set_loaded_dataset_catalog", None
        )
        if callable(set_feature_catalog):
            set_feature_catalog(catalog)

    def _on_dataset_table_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == 0:
            self._notify_dataset_catalog_changed()

    def _on_assembly_branch_dropped(self, branch_name: str, leaf_data: list) -> None:
        """Build the dragged subtree honouring per-node add modes.

        We reach into the source AssemblyTree to recover the originating
        QTreeWidgetItem (the flat `leaf_data` list doesn't carry the
        coherent / incoherent structure). Falls back to the legacy flat
        coherent sum if the item isn't available — e.g. drag from a tree
        we can't introspect.
        """
        from assembly_tree import build_assembly_grid

        panel = getattr(self.assembly_workspace, "assembly_tree_panel", None)
        tree = getattr(panel, "tree", None)
        candidate = getattr(tree, "_branch_drag_item", None)
        branch_item = (
            candidate
            if candidate is not None and candidate.text(0) == branch_name
            else None
        )

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

    def _on_assembly_feature_built(
        self, name: str, payload, history: str
    ) -> None:
        """Publish a completed feature result into GRIM's dataset workflow."""
        if isinstance(payload, RcsGrid):
            self._add_dataset_row(
                payload,
                name,
                history or f"Feature assembly: {name}",
                file_name="",
            )
            self.status.showMessage(f"Feature assembly added: {name}")
            return

        paths: list[str] = []
        if isinstance(payload, (str, os.PathLike)):
            paths = [os.fspath(payload)]
        elif isinstance(payload, (list, tuple)) and all(
            isinstance(value, (str, os.PathLike)) for value in payload
        ):
            paths = [os.fspath(value) for value in payload]
        elif isinstance(payload, dict):
            path_value = payload.get("output_path", payload.get("path"))
            if isinstance(path_value, (str, os.PathLike)):
                paths = [os.fspath(path_value)]

        if paths:
            self._handle_files_dropped(paths)
            return
        self.status.showMessage(
            "Feature build completed, but returned neither an RcsGrid nor "
            "a dataset path."
        )

    def _on_feature_preview_ready(self, plan) -> None:
        """Render only the backend's already-parsed CAD-meter geometry."""
        try:
            self.assembly_workspace.load_feature_preview(plan)
        except Exception as exc:
            self.status.showMessage(f"Feature preview failed: {exc}")
        else:
            stage = str(getattr(plan, "preview_stage", "validated")).lower()
            label = "Input" if stage == "input" else "Validated"
            self.status.showMessage(f"{label} 3-D feature preview updated.")

    def _on_feature_file_built(self, path: str) -> None:
        """Publish a saved feature result through the normal GRIM loader."""
        output = os.fspath(path)
        name = os.path.splitext(os.path.basename(output))[0]
        self.assembly_workspace.publish_feature_build(
            name,
            output,
            "Coherently assembled placed features",
        )

    def _on_ghost_files_exported(self, paths: list, solver_kind: str) -> None:
        """Load GHOST exports through the same path as user-dropped datasets."""
        exported = [
            os.fspath(path)
            for path in paths
            if isinstance(path, (str, os.PathLike))
        ]
        if not exported:
            self.status.showMessage("GHOST export did not contain a dataset path.")
            return
        self.status.showMessage(
            f"Loading {len(exported)} GHOST {str(solver_kind).upper()} export(s)…"
        )
        self._handle_files_dropped(exported)

    def _on_hpc_results_downloaded(self, directory: str) -> None:
        """Load supported datasets found beneath a downloaded HPC result tree."""

        root = Path(os.fspath(directory)).expanduser()
        if not root.is_dir():
            self.status.showMessage(
                f"HPC results downloaded, but the local folder was not found: {root}"
            )
            return
        paths = sorted(
            str(path)
            for path in root.rglob("*")
            if path.is_file() and is_supported_path(str(path))
        )
        if not paths:
            self.status.showMessage(
                f"HPC results downloaded to {root}; no supported datasets were found."
            )
            return
        self.status.showMessage(
            f"Loading {len(paths)} downloaded HPC dataset(s) from {root}…"
        )
        self._handle_files_dropped(paths)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        """Keep the unified app alive while background physics work runs."""
        if self._background_job_active() or bool(
            getattr(self, "_pending_import_batches", ())
        ):
            active_name = self._background_worker_name or "A dataset task"
            QMessageBox.warning(
                self,
                "Dataset Task Still Running",
                f"{active_name} is still running. Wait for it to finish before "
                "closing GRIM.",
            )
            self.main_tabs.setCurrentIndex(0)
            event.ignore()
            return
        if bool(getattr(self, "_isar_busy", False)):
            play_button = getattr(self, "btn_isar_ap_play", None)
            if play_button is not None:
                play_button.setChecked(False)
            current_cancel = getattr(self, "_isar_cancel_event", None)
            if current_cancel is not None:
                current_cancel.set()
            pending = getattr(self, "_isar_pending", None)
            if pending is not None:
                pending_cancel = pending.get("_cancel_event")
                if pending_cancel is not None:
                    pending_cancel.set()
                self._isar_pending = None
            QMessageBox.warning(
                self,
                "ISAR Calculation Still Running",
                "The current ISAR calculation was cancelled. Wait for its "
                "worker to finish, then close GRIM again.",
            )
            self.main_tabs.setCurrentWidget(self.tab_isar)
            event.ignore()
            return
        if self.ppt_workspace.job_is_running():
            operation = self.ppt_workspace.busy_operation() or "PowerPoint report export"
            QMessageBox.warning(
                self,
                "PowerPoint Export Still Running",
                f"{operation} is still running. Wait for it to finish before "
                "closing GRIM.",
            )
            self.main_tabs.setCurrentWidget(self.ppt_workspace)
            self.ppt_workspace.focus_workspace()
            event.ignore()
            return
        feature_busy = bool(
            getattr(self.feature_assembly_panel, "job_is_running", lambda: False)()
        )
        if feature_busy:
            feature_operation = str(
                getattr(
                    self.feature_assembly_panel,
                    "busy_operation",
                    lambda: "",
                )()
            )
            if feature_operation == "build":
                getattr(
                    self.feature_assembly_panel,
                    "request_cancel",
                    lambda: None,
                )()
                detail = (
                    "Safe cancellation was requested. Wait for the current "
                    "numerical step to finish, then close GRIM again. Existing "
                    "output will be kept."
                )
            else:
                detail = (
                    "Feature validation is still running. Wait for it to finish "
                    "before closing GRIM."
                )
            QMessageBox.warning(
                self,
                "Feature Assembly Still Running",
                detail,
            )
            self.main_tabs.setCurrentWidget(self.assembly_workspace)
            event.ignore()
            return
        if self.ghost_integration.solve_is_running():
            QMessageBox.warning(
                self,
                "Solver Still Running",
                "A GHOST solve is still running. Click Cancel in the GHOST "
                "Solver tab, wait for cancellation to finish, and then close GRIM.",
            )
            self.main_tabs.setCurrentWidget(self.ghost_integration)
            self.ghost_integration.focus_solver()
            event.ignore()
            return
        if self.freddy_integration.job_is_running():
            QMessageBox.warning(
                self,
                "FREDDY Task Still Running",
                "A FREDDY material or IBC task is still running. Wait for it "
                "to finish before closing GRIM.",
            )
            self.main_tabs.setCurrentWidget(self.freddy_integration)
            self.freddy_integration.focus_workspace()
            event.ignore()
            return
        if self.runs_workspace.job_is_running():
            operation = self.runs_workspace.busy_operation() or "An HPC operation"
            QMessageBox.warning(
                self,
                "HPC Operation Still Running",
                f"{operation} is still running. Wait for it to finish before "
                "closing GRIM. Submitted SLURM jobs do not require GRIM to stay open.",
            )
            self.main_tabs.setCurrentWidget(self.runs_workspace)
            self.runs_workspace.focus_workspace()
            event.ignore()
            return

        feature_request_close = getattr(
            self.feature_assembly_panel, "request_close", None
        )
        if callable(feature_request_close) and not feature_request_close(self):
            self.main_tabs.setCurrentWidget(self.assembly_workspace)
            event.ignore()
            return

        ghost_workspace = getattr(self.ghost_integration, "workspace", None)
        ghost_request_close = getattr(ghost_workspace, "request_close", None)
        if callable(ghost_request_close) and not ghost_request_close(self):
            self.main_tabs.setCurrentWidget(self.ghost_integration)
            event.ignore()
            return

        assembly_panel = getattr(
            self.assembly_workspace, "assembly_tree_panel", None
        )
        assembly_request_close = getattr(assembly_panel, "request_close", None)
        if callable(assembly_request_close) and not assembly_request_close(self):
            self.main_tabs.setCurrentWidget(self.assembly_workspace)
            event.ignore()
            return

        freddy_workspace = getattr(self.freddy_integration, "workspace", None)
        freddy_request_close = getattr(freddy_workspace, "request_close", None)
        if callable(freddy_request_close) and not freddy_request_close(self):
            self.main_tabs.setCurrentWidget(self.freddy_integration)
            self.freddy_integration.focus_workspace()
            event.ignore()
            return

        if not self._confirm_python_script_close():
            self.main_tabs.setCurrentWidget(self.tab_python)
            event.ignore()
            return

        dirty_rows = self._dirty_dataset_rows()
        if dirty_rows:
            buttons = getattr(QMessageBox, "StandardButton", QMessageBox)
            names = []
            for row in dirty_rows[:10]:
                item = self.table.item(row, 0)
                names.append(item.text() if item is not None else f"Dataset {row + 1}")
            details = "\n".join(f"• {name}" for name in names)
            if len(dirty_rows) > len(names):
                details += f"\n• …and {len(dirty_rows) - len(names)} more"
            answer = QMessageBox.warning(
                self,
                "Unsaved Datasets",
                f"{len(dirty_rows)} derived dataset(s) have not been saved:\n\n"
                f"{details}\n\nSave them before closing?",
                buttons.Save | buttons.Discard | buttons.Cancel,
                buttons.Save,
            )
            if answer == buttons.Cancel:
                event.ignore()
                return
            if answer == buttons.Save:
                directory = QFileDialog.getExistingDirectory(
                    self, "Save Unsaved Datasets Before Closing"
                )
                if not directory or not self._save_rows_to_directory(
                    dirty_rows,
                    directory,
                    dialog_title="Save Unsaved Datasets",
                ):
                    event.ignore()
                    return
        self.runs_workspace.save_settings()
        dispose_ppt = getattr(self.ppt_workspace, "dispose", None)
        if callable(dispose_ppt):
            dispose_ppt()
        super().closeEvent(event)


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
