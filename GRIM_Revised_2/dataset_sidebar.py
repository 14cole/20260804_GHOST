"""Dataset catalog and parameter selection views, without file or math operations.

The shell connects user intentions to its operation controllers. Widget aliases
in the shell preserve the existing integration API while ownership lives here.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QByteArray, QMimeData, Signal
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import (
    QAbstractItemView, QFrame, QGridLayout, QHeaderView, QListWidget, QMenu,
    QScrollArea, QSizePolicy, QSplitter, QTableWidget, QTableWidgetItem,
    QToolButton, QVBoxLayout, QWidget,
)

from assembly_tree import MIME_BRANCH, MIME_DATASET
from grim_widgets import ClickableLabel, CollapsibleSection


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


class DatasetSidebar(QScrollArea):
    """Resizable sidebar shared by Plotting and ISAR."""

    export_pio_requested = Signal()
    export_ptm_requested = Signal()
    export_csv_requested = Signal()
    WIDGET_BINDINGS = (
        "table", "btn_dataset_load", "btn_dataset_save", "btn_dataset_save_all",
        "btn_dataset_export", "btn_dataset_delete", "btn_dataset_undo_delete",
        "btn_dataset_cancel", "list_pol", "list_freq", "list_elev", "list_az",
        "lbl_pol", "lbl_freq", "lbl_elev", "lbl_az", "dataset_parameter_splitter",
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("controlDock")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setMinimumWidth(320)
        dock_body = QWidget()
        dock_body.setObjectName("dockBody")
        dock_layout = QVBoxLayout(dock_body)
        dock_layout.setContentsMargins(8, 8, 8, 8)
        dock_layout.setSpacing(8)

        # ---------- Datasets section (top, grows to fill the dock) ----------
        sec_datasets = CollapsibleSection("Datasets")
        sec_datasets.setObjectName("datasetsSection")
        sec_datasets.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        dataset_actions = QGridLayout()
        dataset_actions.setHorizontalSpacing(4)
        dataset_actions.setVerticalSpacing(4)
        self.btn_dataset_load = QToolButton(text="Load…")
        self.btn_dataset_save = QToolButton(text="Save")
        self.btn_dataset_save_all = QToolButton(text="Save All")
        self.btn_dataset_export = QToolButton(text="Export…")
        self.btn_dataset_export.setPopupMode(QToolButton.InstantPopup)
        dataset_export_menu = QMenu(self.btn_dataset_export)
        action_export_pio = dataset_export_menu.addAction("Pioneer (.pio)…")
        action_export_ptm = dataset_export_menu.addAction("PTM (.ptm)…")
        action_export_csv = dataset_export_menu.addAction("CSV…")
        action_export_pio.triggered.connect(self.export_pio_requested.emit)
        action_export_ptm.triggered.connect(self.export_ptm_requested.emit)
        action_export_csv.triggered.connect(self.export_csv_requested.emit)
        self.btn_dataset_export.setMenu(dataset_export_menu)
        self.btn_dataset_export.setToolTip(
            "Export selected rows to solver interchange (.pio/.ptm) or flat "
            "CSV. Native .grim storage uses Save."
        )
        self.btn_dataset_delete = QToolButton(text="Delete")
        self.btn_dataset_undo_delete = QToolButton(text="Undo Delete")
        self.btn_dataset_undo_delete.setToolTip(
            "Restore the most recently deleted batch of dataset rows, including "
            "their stable identities, unsaved changes, and provenance."
        )
        self.btn_dataset_cancel = QToolButton(text="Cancel Job")
        self.btn_dataset_cancel.setVisible(False)
        self.btn_dataset_cancel.setToolTip(
            "Request cooperative cancellation of the active dataset job. "
            "The current numerical block or file finishes safely before stopping."
        )
        for index, button in enumerate((
            self.btn_dataset_load, self.btn_dataset_save, self.btn_dataset_save_all,
            self.btn_dataset_export, self.btn_dataset_delete, self.btn_dataset_undo_delete,
        )):
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            dataset_actions.addWidget(button, index // 3, index % 3)
        dataset_actions.addWidget(self.btn_dataset_cancel, 2, 0, 1, 3)
        for column in range(3):
            dataset_actions.setColumnStretch(column, 1)
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
        sec_params.setObjectName("parametersSection")
        sec_params.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        params_grid = QGridLayout()
        params_grid.setHorizontalSpacing(6)
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
            widget.setMinimumWidth(0)
            widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
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
            lbl.setWordWrap(True)
            lbl.setMinimumWidth(0)
            lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
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

        # The vertical handle directly controls the balance between the
        # dataset table and parameter lists. Moving it upward gives Parameters
        # more height; moving it downward gives Datasets more height.
        self.dataset_parameter_splitter = QSplitter(Qt.Vertical)
        self.dataset_parameter_splitter.setObjectName(
            "datasetParameterSplitter"
        )
        self.dataset_parameter_splitter.setChildrenCollapsible(False)
        self.dataset_parameter_splitter.setHandleWidth(8)
        self.dataset_parameter_splitter.addWidget(sec_datasets)
        self.dataset_parameter_splitter.addWidget(sec_params)
        self.dataset_parameter_splitter.setStretchFactor(0, 3)
        self.dataset_parameter_splitter.setStretchFactor(1, 2)
        self.dataset_parameter_splitter.setSizes([480, 260])
        dock_layout.addWidget(self.dataset_parameter_splitter, 1)
        self.setWidget(dock_body)
