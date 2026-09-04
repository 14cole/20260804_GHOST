"""Placement authoring with explicit units, bounded history and atomic CSV saves."""
from __future__ import annotations
import csv
import io
import math
import os
from pathlib import Path
import tempfile
import threading

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox, QInputDialog,
    QAbstractItemView, QDialogButtonBox)

MAX_ROWS = 10000
MAX_HISTORY_BYTES = 16*1024**2


def unique_id(prefix, existing):
    if prefix not in existing:
        return prefix
    index = 2
    while f"{prefix}_{index}" in existing:
        index += 1
    return f"{prefix}_{index}"


def point_array(seed, existing_ids, *, count, step=None, radius=None):
    """Generate a row or a full circle using the seed's local +x/+y frame."""
    count = int(count)
    if not 1 <= count <= MAX_ROWS:
        raise ValueError(f"Count must be 1–{MAX_ROWS}.")
    origin = np.asarray(seed[2:5], float)
    normal = np.asarray(seed[5:8], float)
    roll = np.asarray(seed[8:11], float)
    if not np.all(np.isfinite(np.r_[origin, normal, roll])) or np.linalg.norm(normal) < 1e-12:
        raise ValueError("Seed coordinates and orientation must be finite and its normal nonzero.")
    normal /= np.linalg.norm(normal)
    roll -= (roll@normal)*normal
    if np.linalg.norm(roll) < 1e-12:
        raise ValueError("Seed roll reference must not be parallel to its normal.")
    roll /= np.linalg.norm(roll)
    used = set(existing_ids) - {seed[0]}
    next_suffix = 2
    result = []
    for index in range(count):
        if radius is None:
            offset = index*np.asarray(step, float)
        else:
            if not math.isfinite(float(radius)) or float(radius) <= 0:
                raise ValueError("Circle radius must be positive and finite.")
            angle = 2*math.pi*index/count
            offset = float(radius)*(math.cos(angle)*roll+math.sin(angle)*np.cross(normal, roll))
        position = origin+offset
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("Row spacing must contain three finite coordinates.")
        row = list(seed)
        identifier = seed[0]
        while identifier in used:
            identifier = f"{seed[0]}_{next_suffix}"
            next_suffix += 1
        row[0] = identifier
        used.add(row[0])
        row[2:5] = [f"{value:.15g}" for value in position]
        result.append(row)
    return result


def project_to_surface(points_m, *, surface=None, profile=None, cancel_check=lambda: False):
    """Nearest skin points and normals in CAD coordinates, in bounded batches."""
    points = np.asarray(points_m, float).reshape(-1, 3)
    if not np.all(np.isfinite(points)):
        raise ValueError("Coordinates must be finite.")
    nearest, normals = np.empty_like(points), np.empty_like(points)
    if surface is None and profile is None:
        raise ValueError("Choose a matching surface mesh or an embedded BoR body first.")
    for start in range(0, len(points), 128):
        if cancel_check():
            raise InterruptedError("Surface projection cancelled.")
        sl = slice(start, start+128)
        if surface is not None:
            _, nearest[sl], normals[sl], _ = surface.nearest(points[sl])
        else:
            # CAD radial coordinates are x/right,z/up; y is the profile axis.
            local = points[sl]
            rho = np.hypot(local[:, 0], local[:, 2])
            radial = np.column_stack((local[:, 0], local[:, 2]))/np.maximum(rho[:, None], 1e-280)
            radial[rho < 1e-12] = [0., 1.]
            q = np.column_stack((rho, local[:, 1]))
            p0, delta = np.asarray(profile[:-1], float), np.diff(np.asarray(profile, float), axis=0)
            lengths = np.linalg.norm(delta, axis=1)
            if np.any(lengths <= 0):
                raise ValueError("Body profile has a zero-length segment.")
            fraction = np.clip(np.sum((q[:, None]-p0)*delta, axis=-1)/lengths**2, 0, 1)
            feet = p0+fraction[:, :, None]*delta
            index = np.argmin(np.sum((feet-q[:, None])**2, axis=-1), axis=1)
            foot = feet[np.arange(len(local)), index]
            n = np.column_stack((-delta[index, 1], delta[index, 0]))/lengths[index, None]
            nearest[sl] = np.column_stack((foot[:, 0]*radial[:, 0], foot[:, 1], foot[:, 0]*radial[:, 1]))
            normals[sl] = np.column_stack((n[:, 0]*radial[:, 0], n[:, 1], n[:, 0]*radial[:, 1]))
    return nearest, normals


def point_path(seed, existing_ids, vertices, count):
    """Place uniformly by arclength; closed paths omit a duplicate endpoint."""
    vertices = np.asarray(vertices, float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 2 or not np.all(np.isfinite(vertices)):
        raise ValueError("Path requires at least two finite x,y,z vertices.")
    lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    if np.any(lengths <= 0):
        raise ValueError("Consecutive path vertices must be distinct.")
    distance = np.r_[0., np.cumsum(lengths)]
    closed = np.array_equal(vertices[0], vertices[-1])
    rows = point_array(seed, existing_ids, count=count, step=[0., 0., 0.])
    samples = np.linspace(0., distance[-1], len(rows), endpoint=not closed)
    positions = np.column_stack([np.interp(samples, distance, vertices[:, axis]) for axis in range(3)])
    for row, position in zip(rows, positions):
        row[2:5] = [f"{value:.15g}" for value in position]
    spacing = distance[-1]/max(1, len(rows) if closed else len(rows)-1)
    return rows, float(distance[-1]), float(spacing)


class _SurfaceWorker(QObject):
    done = Signal(object)
    failed = Signal(str)
    def __init__(self, task):
        super().__init__()
        self.task = task
    def run(self):
        try:
            self.done.emit(self.task())
        except Exception as exc:
            self.failed.emit(str(exc))


class PlacementEditor(QDialog):
    instance_selected = Signal(str, str)
    saved = Signal(str)

    def __init__(self, kind, *, columns, units, path="", base_dir=None,
                 surface_loader=None, validator=None, parent=None):
        super().__init__(parent)
        self.kind, self.columns, self.units = kind, tuple(columns), units
        self.path = Path(path) if path else None
        if self.path is not None and not self.path.is_absolute():
            self.path = Path(base_dir or os.getcwd())/self.path
        self.surface_loader, self.validator = surface_loader, validator
        self._history, self._future, self._loading = [], [], False
        self._thread = None
        self._cancel = threading.Event()
        self.setWindowTitle(f"Edit {kind} placements")
        self.resize(1100 if kind == "point" else 1320, 650)
        layout = QVBoxLayout(self)
        note = QLabel(f"Coordinates: {units}. CAD +x right, +y forward, +z up. Normal and roll vectors are unitless. Changes stay in this editor until Save. Surface projection changes phase locations only when Snap is selected.")
        note.setWordWrap(True)
        layout.addWidget(note)
        actions = QHBoxLayout()
        self.buttons = {}
        for label, callback in (("Add", self.add), ("Duplicate", self.duplicate), ("Delete", self.delete),
                ("Undo", self.undo), ("Redo", self.redo),
                ("Point row…" if kind == "point" else "Path vertices…", self.generate),
                ("Point circle…", self.circle), ("Points on path…", self.along_path), ("Derive normals", lambda: self.project(False)),
                ("Snap + normals", lambda: self.project(True))):
            if kind != "point" and label in {"Point circle…", "Points on path…"}:
                continue
            button = QPushButton(label)
            button.clicked.connect(callback)
            self.buttons[label] = button
            actions.addWidget(button)
        layout.addLayout(actions)
        self.table = QTableWidget(0, len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.horizontalHeader().setMinimumSectionSize(62)
        self.table.horizontalHeader().setDefaultSectionSize(78)
        self.table.setColumnWidth(0, 125)
        self.table.setColumnWidth(1, 125)
        if kind == "line":
            self.table.setColumnWidth(2, 110)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.itemChanged.connect(self._edited)
        self.table.itemSelectionChanged.connect(self._selected)
        layout.addWidget(self.table, 1)
        self.status = QLabel("Save validates the exact CSV schema. Physical skin and library checks run in Assembly.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.dialog_buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.dialog_buttons.accepted.connect(self.save)
        self.dialog_buttons.rejected.connect(self.reject)
        layout.addWidget(self.dialog_buttons)
        rows = []
        if self.path and self.path.is_file():
            if self.path.stat().st_size > 16*1024**2:
                raise ValueError("Placement editor accepts CSV files up to 16 MiB; split this input into smaller authored groups.")
            with self.path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream)
                if tuple(next(reader, ())) != self.columns:
                    raise ValueError("CSV header does not match the current placement schema.")
                rows = list(reader)
        self._replace(rows)
        self._last = self.rows()
        self._saved_rows = self.rows()

    def rows(self):
        return [[self.table.item(r, c).text() if self.table.item(r, c) else "" for c in range(len(self.columns))] for r in range(self.table.rowCount())]

    def _replace(self, rows):
        if len(rows) > MAX_ROWS or any(len(row) != len(self.columns) for row in rows):
            raise ValueError(f"Editor requires matching columns and at most {MAX_ROWS:,} rows.")
        self._loading = True
        try:
            self.table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                for c, value in enumerate(row):
                    self.table.setItem(r, c, QTableWidgetItem(str(value)))
        finally:
            self._loading = False

    def _remember(self):
        self._history.append(self._last)
        self._future.clear()
        while len(self._history) > 1 and (len(self._history) > 30 or sum(len(repr(rows))*2 for rows in self._history) > MAX_HISTORY_BYTES):
            self._history.pop(0)

    def _edited(self, *_):
        if not self._loading:
            self._remember()
            self._last = self.rows()

    def change(self, rows):
        if len(rows) > MAX_ROWS:
            raise ValueError(f"Editor limit is {MAX_ROWS:,} rows.")
        self._remember()
        self._replace(rows)
        self._last = self.rows()

    def selected_rows(self):
        return sorted({item.row() for item in self.table.selectedItems()})

    def _selected(self):
        selected = self.selected_rows()
        if selected and not self._loading:
            self.instance_selected.emit(self.kind, self.rows()[selected[0]][0])

    def add(self):
        rows = self.rows()
        identifier = unique_id(self.kind, {row[0] for row in rows})
        seed = [identifier, "feature", "0", "0", "0", "0", "0", "1", "1", "0", "0"] if self.kind == "point" else [identifier, "feature", "1", "0", "0", "0", "1", "0", "0", "0", "0", "1", "0", "0", "1"]
        self._run(lambda: self.change(rows+[seed]))

    def duplicate(self):
        def action():
            rows = self.rows()
            selected_ids = {rows[i][0] for i in self.selected_rows()}
            source = [rows[i] for i in self.selected_rows()] if self.kind == "point" else [row for row in rows if row[0] in selected_ids]
            used, names = {row[0] for row in rows}, {}
            copies = []
            for row in source:
                copy = list(row)
                if row[0] not in names:
                    names[row[0]] = unique_id(row[0], used)
                    used.add(names[row[0]])
                copy[0] = names[row[0]]
                copies.append(copy)
            self.change(rows+copies)
            self.status.setText("Duplicated at the same coordinates. Edit the new positions before physical validation.")
        self._run(action)

    def delete(self):
        selected = set(self.selected_rows())
        self.change([row for index, row in enumerate(self.rows()) if index not in selected])

    def undo(self):
        if self._history:
            self._future.append(self.rows())
            self._replace(self._history.pop())
            self._last = self.rows()

    def redo(self):
        if self._future:
            self._history.append(self.rows())
            self._replace(self._future.pop())
            self._last = self.rows()

    def _run(self, action):
        try:
            action()
        except Exception as exc:
            self.status.setText(str(exc))

    def generate(self):
        self._run(lambda: self._generate(False))

    def circle(self):
        self._run(lambda: self._generate(True))

    def along_path(self):
        def action():
            selected, rows = self.selected_rows(), self.rows()
            if len(selected) != 1:
                raise ValueError("Select one seed row for a point path.")
            index = selected[0]
            text, accepted = QInputDialog.getMultiLineText(self, "Point path", f"One x,y,z vertex per line in {self.units}. Repeat the first vertex to close. Points inherit the seed normal and roll; derive surface normals afterward.")
            if not accepted:
                return
            vertices = [[float(value) for value in line.replace(",", " ").split()] for line in text.splitlines() if line.strip()]
            count, accepted = QInputDialog.getInt(self, "Point path count", "Number of points replacing the seed:", 5, 1, MAX_ROWS)
            if not accepted:
                return
            generated, length, spacing = point_path(rows[index], {row[0] for row in rows}, vertices, count)
            self.change(rows[:index]+generated+rows[index+1:])
            self.status.setText(f"{count} points; path length {length:g} {self.units}; arclength spacing {spacing:g} {self.units}. Inspect before Save; Undo restores the seed.")
        self._run(action)

    def _generate(self, circle):
        selection, rows = self.selected_rows(), self.rows()
        if len(selection) != 1:
            raise ValueError("Select one seed row to generate a pattern or replace its path.")
        index, seed = selection[0], rows[selection[0]]
        if self.kind == "line":
            text, accepted = QInputDialog.getMultiLineText(self, "Ordered path vertices", f"One x,y,z vertex per line, in {self.units}. Replaces the whole selected line_id; inherits its first endpoint normal.")
            if not accepted:
                return
            vertices = np.asarray([[float(v) for v in line.replace(",", " ").split()] for line in text.splitlines() if line.strip()])
            if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 2 or not np.all(np.isfinite(vertices)):
                raise ValueError("Provide at least two finite x,y,z vertices.")
            generated = [[seed[0], seed[1], str(i+1), *map(str, a), *map(str, b), *seed[9:12], *seed[9:12]] for i, (a, b) in enumerate(zip(vertices[:-1], vertices[1:]))]
            self.change([row for row in rows if row[0] != seed[0]]+generated)
            self.status.setText(f"{len(generated)} segments; path length {np.linalg.norm(np.diff(vertices, axis=0), axis=1).sum():g} {self.units}. Derive normals, inspect, then Save.")
        else:
            count, accepted = QInputDialog.getInt(self, "Pattern count", "Number of points (replaces the seed):", 5, 1, MAX_ROWS)
            if not accepted:
                return
            if circle:
                radius, accepted = QInputDialog.getDouble(self, "Circle radius", f"Radius in {self.units}; seed position is circle center:", 1., .000001, 1e9, 6)
                options = {"radius": radius}
            else:
                spacing, accepted = QInputDialog.getText(self, "Row spacing", f"Step dx,dy,dz in {self.units}:")
                options = {"step": [float(v) for v in spacing.replace(",", " ").split()]} if accepted else {}
            if accepted:
                generated = point_array(seed, {row[0] for row in rows}, count=count, **options)
                self.change(rows[:index]+generated+rows[index+1:])

    def project(self, snap):
        def start():
            if self._thread is not None:
                return
            rows, selected = self.rows(), self.selected_rows()
            if not selected or self.surface_loader is None:
                raise ValueError("Select rows and choose a body surface first.")
            self._cancel.clear()
            def work():
                surface, profile, scale = self.surface_loader()
                points = [rows[i][2:5] for i in selected] if self.kind == "point" else [v for i in selected for v in (rows[i][3:6], rows[i][6:9])]
                original = np.asarray(points, float)*scale
                nearest, normals = project_to_surface(original, surface=surface, profile=profile, cancel_check=self._cancel.is_set)
                for j, index in enumerate(selected):
                    if self.kind == "point":
                        if snap:
                            rows[index][2:5] = list(map(str, nearest[j]/scale))
                        rows[index][5:8] = list(map(str, normals[j]))
                    else:
                        if snap:
                            rows[index][3:9] = list(map(str, (nearest[2*j:2*j+2]/scale).ravel()))
                        rows[index][9:15] = list(map(str, normals[2*j:2*j+2].ravel()))
                return rows, float(np.max(np.linalg.norm(nearest-original, axis=1)))
            self._thread = QThread(self)
            self._worker = _SurfaceWorker(work)
            self._worker.moveToThread(self._thread)
            self._thread.started.connect(self._worker.run)
            self._worker.done.connect(self._projection_done)
            self._worker.failed.connect(self.status.setText)
            self._worker.done.connect(self._thread.quit)
            self._worker.failed.connect(self._thread.quit)
            self._thread.finished.connect(self._worker.deleteLater)
            self._thread.finished.connect(self._projection_finished)
            self.table.setEnabled(False)
            for button in self.buttons.values():
                button.setEnabled(False)
            self.dialog_buttons.button(QDialogButtonBox.Save).setEnabled(False)
            self.status.setText("Finding nearest body surface… Cancel stops after the current bounded query.")
            self._thread.start()
        self._run(start)

    def _projection_done(self, result):
        if not self._cancel.is_set():
            self.change(result[0])
            self.status.setText(f"Surface helper complete; maximum original offset {result[1]*1000:.6g} mm. Check roll directions and run physical validation. Undo restores the authored values.")

    def _projection_finished(self):
        self._thread.deleteLater()
        self._thread = None
        self.table.setEnabled(True)
        for button in self.buttons.values():
            button.setEnabled(True)
        self.dialog_buttons.button(QDialogButtonBox.Save).setEnabled(True)

    def save(self):
        def action():
            path, _ = QFileDialog.getSaveFileName(self, "Save placement CSV", str(self.path or f"{self.kind}_placements.csv"), "Placement CSV (*.csv)")
            if not path:
                return
            destination = Path(path)
            if destination.suffix.lower() != ".csv":
                destination = destination.with_suffix(".csv")
            descriptor, temporary = tempfile.mkstemp(prefix=".placement-", suffix=".csv", dir=destination.parent)
            try:
                with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(self.columns)
                    writer.writerows(self.rows())
                if self.validator is not None:
                    self.validator(temporary)
                os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            self.path = destination
            self._saved_rows = self.rows()
            self.saved.emit(str(destination))
            self.accept()
        self._run(action)

    def reject(self):
        if self._thread is not None:
            self._cancel.set()
            self.status.setText("Cancelling surface helper; close again when it has stopped.")
            return
        if self.rows() != self._saved_rows and QMessageBox.question(self, "Discard edits?", "Close without saving placement edits?", QMessageBox.Discard | QMessageBox.Cancel) != QMessageBox.Discard:
            return
        super().reject()

    def closeEvent(self, event):
        if self._thread is not None:
            self._cancel.set()
            event.ignore()
        else:
            event.ignore()
            self.reject()
