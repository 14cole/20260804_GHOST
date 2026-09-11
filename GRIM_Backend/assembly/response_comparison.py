"""In-tab body/feature/total cuts with bounded archive reads."""
from pathlib import Path
import json
import threading
import zipfile
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QFileDialog, QCheckBox
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT


def response_axes(path):
    with np.load(path, allow_pickle=False) as archive:
        try:
            units = json.loads(str(np.asarray(archive["units"]).item()))
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"{Path(path).name}: comparison requires declared 3-D RCS units.") from exc
        required = {"azimuth": "deg", "elevation": "deg", "frequency": "GHz", "rcs_linear_quantity": "sigma_3d", "rcs_log_unit": "dBsm"}
        if not isinstance(units, dict) or any(units.get(key) != value for key, value in required.items()):
            raise ValueError(f"{Path(path).name}: comparison requires sigma_3d / dBsm, degree angles and GHz frequencies.")
        axes = tuple(np.asarray(archive[key]) for key in ("azimuths", "elevations", "frequencies", "polarizations"))
    for axis in axes:
        if axis.ndim != 1 or not len(axis) or len(np.unique(axis)) != len(axis):
            raise ValueError("Comparison axes must be nonempty, unique 1-D samples.")
    for axis in axes[:3]:
        if not np.all(np.isfinite(axis.astype(float))) or np.any(np.diff(axis.astype(float)) <= 0):
            raise ValueError("Comparison angular and frequency axes must be finite and increasing.")
    return axes


def read_response_cut(path, *, frequency, elevation, polarization, cancel_check=lambda: False):
    """Stream only the selected values from compressed NPY, without a grid copy."""
    az, el, frequencies, pol = response_axes(path)
    def exact(axis, value):
        matches = np.flatnonzero(np.asarray(axis, str) == str(value)) if isinstance(value, str) else np.flatnonzero(np.isclose(axis.astype(float), value, rtol=0, atol=1e-10))
        if len(matches) != 1:
            raise ValueError(f"{Path(path).name}: cut sample {value!r} is not stored exactly once.")
        return int(matches[0])
    ei, fi, pi = exact(el, elevation), exact(frequencies, frequency), exact(pol, polarization)
    with zipfile.ZipFile(path) as archive, archive.open("rcs_power.npy") as stream:
        version = np.lib.format.read_magic(stream)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(stream)
        if tuple(shape) != (len(az), len(el), len(frequencies), len(pol)) or dtype.kind not in "fiu":
            raise ValueError("Response power array does not match numeric 4-D axes.")
        data_start = stream.tell()
        values = np.empty(len(az), float)
        for ai in range(len(az)):
            if cancel_check():
                raise InterruptedError("Response comparison cancelled.")
            index = int(np.ravel_multi_index((ai, ei, fi, pi), shape, order="F" if fortran else "C"))
            offset = data_start+index*dtype.itemsize
            # Explicit small reads avoid allocating the decompressed gap even
            # for long elevation/frequency axes in a C-order response cube.
            while stream.tell() < offset:
                if cancel_check():
                    raise InterruptedError("Response comparison cancelled.")
                remaining = offset-stream.tell()
                if not stream.read(min(256*1024, remaining)):
                    raise ValueError("Truncated response power array.")
            raw = stream.read(dtype.itemsize)
            if len(raw) != dtype.itemsize:
                raise ValueError("Truncated response power array.")
            values[ai] = np.frombuffer(raw, dtype=dtype, count=1)[0]
    if np.any(np.isinf(values)) or np.any(values < 0):
        raise ValueError("Response cut contains infinite or negative RCS power.")
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10*np.log10(np.where(values >= 0, values, np.nan))
    return az, db


class _CutWorker(QObject):
    done = Signal(object)
    def __init__(self, paths, selected, cancel):
        super().__init__()
        self.paths, self.selected, self.cancel = paths, selected, cancel
    @Slot()
    def run(self):
        result = {"curves": [], "errors": []}
        try:
            axes = response_axes(self.paths[-1][1])
            selection = self.selected or (float(axes[2][0]), float(axes[1][np.argmin(np.abs(axes[1]))]), str(axes[3][0]))
            result.update(axes=axes, selected=selection)
            for label, path in self.paths:
                if self.cancel.is_set():
                    return
                try:
                    az, db = read_response_cut(path, frequency=selection[0], elevation=selection[1], polarization=selection[2], cancel_check=self.cancel.is_set)
                    result["curves"].append((label, az, db))
                except Exception as exc:
                    result["errors"].append(str(exc))
        except Exception as exc:
            result["errors"].append(str(exc))
        finally:
            self.done.emit(result)


class ResponseComparison(QWidget):
    ready = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self._paths, self._thread, self._pending = [], None, None
        self._cancel = threading.Event()
        self._last_result, self._difference_axes = None, None
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.frequency, self.elevation, self.polarization = QComboBox(), QComboBox(), QComboBox()
        for name, combo in (("GHz", self.frequency), ("Elevation", self.elevation), ("Pol", self.polarization)):
            controls.addWidget(QLabel(name))
            controls.addWidget(combo)
            combo.currentIndexChanged.connect(self.refresh)
        add = QPushButton("Add saved variant / family…")
        add.clicked.connect(self.add_variant)
        controls.addWidget(add)
        controls.addStretch()
        layout.addLayout(controls)
        self.source_label = QLabel("Comparison uses saved response files.")
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)
        self.show_difference = QCheckBox("Show total minus body RCS (dB) on right axis")
        self.show_difference.toggled.connect(self._redraw_difference)
        layout.addWidget(self.show_difference)
        self.figure = Figure(figsize=(8, 5), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas, 1)
        self.status = QLabel("Build an Assembly to compare body, features and coherent total. Double-click a feature in the 3-D tab to select it for editing.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def set_outputs(self, body, features, total):
        self._paths = [(label, str(path)) for label, path in (("Body", body), ("Features only", features), ("Coherent total", total)) if path]
        self.source_label.setText(f"Saved build: {Path(total).name}. Curves update after a new build or cut selection.")
        self.source_label.setToolTip("\n".join(f"{label}: {path}" for label, path in self._paths))
        self._queue(None)

    def refresh(self, *_):
        if self._paths and all(combo.count() for combo in (self.frequency, self.elevation, self.polarization)):
            self._queue((float(self.frequency.currentData()), float(self.elevation.currentData()), str(self.polarization.currentData())))

    def _queue(self, selection):
        self._pending = (list(self._paths), selection)
        if self._thread is not None:
            self._cancel.set()
            return
        paths, selection = self._pending
        self._pending = None
        self._cancel = threading.Event()
        self._thread = QThread(self)
        self._worker = _CutWorker(paths, selection, self._cancel)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._show)
        self._worker.done.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._stopped)
        self.status.setText("Reading exact response cut…")
        self._thread.start()

    def _show(self, result):
        if self._cancel.is_set():
            return
        self._last_result = result
        if "axes" in result:
            az, el, frequencies, pol = result["axes"]
            for combo, values, selected in zip((self.frequency, self.elevation, self.polarization), (frequencies, el, pol), result["selected"]):
                combo.blockSignals(True)
                combo.clear()
                for value in values:
                    item = value.item() if isinstance(value, np.generic) else value
                    combo.addItem(str(item), item)
                combo.setCurrentIndex(max(0, combo.findData(selected)))
                combo.blockSignals(False)
        if self._difference_axes is not None:
            self._difference_axes.remove()
            self._difference_axes = None
        self.axes.clear()
        for label, az, db in result["curves"]:
            self.axes.plot(az, db, label=label)
        self.axes.set(xlabel="Azimuth (deg)", ylabel="RCS (dBsm)")
        self.axes.grid(alpha=.25)
        if result["curves"]:
            self.axes.legend()
        if self.show_difference.isChecked():
            curves = {label: (az, db) for label, az, db in result["curves"]}
            if "Body" in curves and "Coherent total" in curves:
                body_az, body_db = curves["Body"]
                total_az, total_db = curves["Coherent total"]
                az, bi, ti = np.intersect1d(body_az, total_az, return_indices=True)
                valid = np.isfinite(body_db[bi]) & np.isfinite(total_db[ti])
                self._difference_axes = self.axes.twinx()
                self._difference_axes.plot(az[valid], total_db[ti[valid]]-body_db[bi[valid]], color="purple", linestyle="--", alpha=.7)
                self._difference_axes.set_ylabel("Total − body RCS (dB)", color="purple")
        self.canvas.draw_idle()
        self.status.setText(" | ".join(result["errors"]) or "Exact stored samples. Features-only RCS is 4π|ΔF|²; total RCS includes coherent interference with the body. Save family-only variants with the feature Use checkboxes, then add their saved responses here.")
        self.ready.emit()

    def _redraw_difference(self, *_):
        if self._last_result is not None:
            self._show(self._last_result)

    def _stopped(self):
        self._thread.deleteLater()
        self._thread = None
        if self._pending is not None:
            paths, selection = self._pending
            self._paths = paths
            self._queue(selection)

    def add_variant(self):
        path, _ = QFileDialog.getOpenFileName(self, "Add saved comparison", "", "GRIM response (*.grim)")
        if path:
            if len(self._paths) >= 10:
                self.status.setText("Up to ten curves per comparison. Build again to reset the comparison list.")
                return
            self._paths.insert(max(0, len(self._paths)-1), (Path(path).stem, path))
            self.refresh() if self.frequency.count() else self._queue(None)

    def can_close(self):
        if self._thread is not None:
            self._pending = None
            self._cancel.set()
            return False
        return True
