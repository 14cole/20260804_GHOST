"""Interactive, read-only feature interference and gain/phase sensitivity."""
import threading
import json
from types import SimpleNamespace
import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from ghost_integration import load_ghost_module


class _InspectorWorker(QObject):
    done = Signal(object)
    def __init__(self, service, plan, sample, cancel):
        super().__init__()
        self.service, self.plan, self.sample, self.cancel = service, plan, sample, cancel

    @Slot()
    def run(self):
        try:
            value = self.service.evaluate(self.plan, *self.sample, cancel_check=self.cancel.is_set)
            self.done.emit(value)
        except Exception as exc:
            self.done.emit({"error": str(exc)})


class InterferenceInspector(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plan_provider = lambda: None
        self._service = self._thread = self._result = None
        self._cancel = threading.Event()
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.frequency, self.azimuth, self.elevation, self.polarization = (QComboBox() for _ in range(4))
        for label, combo in (("GHz", self.frequency), ("Azimuth", self.azimuth), ("Elevation", self.elevation), ("Pol", self.polarization)):
            controls.addWidget(QLabel(label))
            controls.addWidget(combo)
        self.polarization.addItems(["VV", "HH", "VH"])
        self.polarization.currentIndexChanged.connect(self._recalculate)
        self.evaluate = QPushButton("Inspect validated assembly")
        self.evaluate.clicked.connect(self._evaluate)
        controls.addWidget(self.evaluate)
        layout.addLayout(controls)
        self.family_study = QPushButton("Check corner / termination / curvature / pair study...")
        self.family_study.clicked.connect(self._check_family_study)
        layout.addWidget(self.family_study)
        self.status = QLabel("Validate the Assembly, then inspect an exact stored sample. Gain, phase, and Use edits below are previews at fixed geometry; they do not edit the Assembly.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(["Use", "Feature", "Gain", "Phase offset (deg)", "|F| (m)", "Phase vs rest (deg)", "Interference (m2)", "Removal effect (m2)", "d(sigma)/d(gain)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.itemChanged.connect(self._recalculate)
        layout.addWidget(self.table, 1)
        self.figure = Figure(figsize=(5, 3), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(210)
        self.axes = self.figure.add_subplot(111)
        self._palette = {"panel_bg": "#ffffff", "text": "#1f2937", "grid": "#cbd5e1", "border": "#cbd5e1"}
        layout.addWidget(self.canvas, 1)
        self.total = QLabel("")
        self.total.setWordWrap(True)
        layout.addWidget(self.total)
        self._plan_timer = QTimer(self)
        self._plan_timer.setInterval(500)
        self._plan_timer.timeout.connect(self._check_current_plan)
        self._plan_timer.start()

    def apply_application_palette(self, palette):
        self._palette = dict(palette)
        self._style_plot()
        self.canvas.draw_idle()

    def _style_plot(self):
        palette = self._palette
        background, foreground = palette["panel_bg"], palette["text"]
        self.figure.set_facecolor(background)
        self.axes.set_facecolor(background)
        self.axes.tick_params(colors=foreground)
        self.axes.xaxis.label.set_color(foreground)
        self.axes.yaxis.label.set_color(foreground)
        for spine in self.axes.spines.values():
            spine.set_color(palette["border"])
        self.axes.grid(color=palette["grid"], alpha=.3)
        legend = self.axes.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(background)
            legend.get_frame().set_edgecolor(palette["border"])
            for label in legend.get_texts():
                label.set_color(foreground)

    def _check_current_plan(self):
        if self._result is None:
            return True
        plan = self.plan_provider()
        if plan is None or getattr(plan, "prepared_plan_sha256", None) != self._result["key"][0]:
            self._show({"error": "Assembly changed. Validate it again and inspect a new sample."})
            return False
        return True

    def _evaluate(self):
        if self._thread is not None:
            return
        plan = self.plan_provider()
        if plan is None:
            self.status.setText("Validate the current Assembly first, then inspect it here.")
            return
        for combo, key in ((self.frequency, "frequencies_ghz"), (self.azimuth, "azimuths_deg"), (self.elevation, "elevations_deg")):
            current = combo.currentData()
            combo.clear()
            for value in plan.radar_grid[key]:
                combo.addItem(f"{float(value):g}", float(value))
            index = combo.findData(current)
            combo.setCurrentIndex(index if index >= 0 else 0)
        try:
            if self._service is None:
                self._service = load_ghost_module("assembly_inspector").ContributionInspector()
        except Exception as exc:
            self.status.setText(str(exc))
            return
        sample = tuple(float(combo.currentData()) for combo in (self.frequency, self.azimuth, self.elevation))
        self._launch(self._service, plan, sample, self._show)

    def _launch(self, service, plan, sample, receiver):
        self._cancel = threading.Event()
        self._thread = QThread(self)
        self._worker = _InspectorWorker(service, plan, sample, self._cancel)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(receiver)
        self._worker.done.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._stopped)
        self.evaluate.setEnabled(False)
        self.family_study.setEnabled(False)
        self.status.setText("Checking the validated sources and evaluating each complex feature contribution...")
        self._thread.start()

    def _check_family_study(self):
        if self._thread is not None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Feature family study definition", "", "Study JSON (*.json)")
        if not path:
            return
        try:
            function = load_ghost_module("feature_family_validation").validate_study
        except Exception as exc:
            self.status.setText(str(exc))
            return
        self._launch(SimpleNamespace(evaluate=function), path, (), self._show_family_study)
        self.status.setText("Checking family references, mesh refinements, and complex reconstruction errors...")

    def _show_family_study(self, result):
        if self._cancel.is_set():
            return
        if "error" in result:
            self.status.setText(result["error"])
            return
        message = "\n".join(f"{row['id']}: {row['status']}" for row in result["cases"])
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Feature family evidence")
        dialog.setText(message+"\n\n"+result["interpretation"])
        dialog.setDetailedText(json.dumps(result, indent=2))
        dialog.exec()
        self.status.setText(f"Family study: {sum(row['passed'] for row in result['cases'])}/{len(result['cases'])} cases passed. Missing references remain unvalidated.")

    def _show(self, result):
        if self._cancel.is_set():
            return
        if "error" in result:
            self._result = None
            self.table.setRowCount(0)
            self.axes.clear()
            self._style_plot()
            self.canvas.draw_idle()
            self.total.clear()
            self.status.setText(result["error"])
            return
        self._result = result
        if not self._check_current_plan():
            return
        self.table.blockSignals(True)
        self.table.setRowCount(len(result["labels"]))
        for row, label in enumerate(result["labels"]):
            for col, text in enumerate(("", label, "1", "0", "", "", "", "", "")):
                item = QTableWidgetItem(text)
                if col not in (2, 3):
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if col == 0:
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Checked)
                self.table.setItem(row, col, item)
        self.table.blockSignals(False)
        self.status.setText("Toggle Use or edit gain/phase to preview interference instantly. Positive removal effect means this feature raises the total RCS. Negative interference means cancellation. This model does not add mutual coupling.")
        self._recalculate()

    def _recalculate(self, *_):
        if self._result is None or not self._check_current_plan():
            return
        try:
            if any(float(self.table.item(r, 2).text()) < 0 for r in range(self.table.rowCount())):
                raise ValueError("Gain must be nonnegative; use phase for a sign reversal.")
            gains = np.array([float(self.table.item(r, 2).text()) * np.exp(1j*np.radians(float(self.table.item(r, 3).text()))) if self.table.item(r, 0).checkState() == Qt.Checked else 0j for r in range(self.table.rowCount())])
            p = self.polarization.currentIndex()
            body, fields = self._result["body"][p], self._result["fields"][:, p]
            metrics = load_ghost_module("assembly_inspector").interference_metrics(body, fields, gains)
        except (ValueError, TypeError) as exc:
            self.total.setText(f"Enter finite numeric gains and phase offsets: {exc}")
            return
        self.table.blockSignals(True)
        for row in range(len(fields)):
            enabled = self.table.item(row, 0).checkState() == Qt.Checked
            derivative = metrics["gain_derivative_m2"][row] if enabled and abs(gains[row]) > 0 else np.nan
            values = (abs(metrics["applied"][row]), metrics["relative_phase_deg"][row], metrics["interference_m2"][row], metrics["removal_change_m2"][row], derivative)
            for col, value in enumerate(values, 4):
                self.table.item(row, col).setText(f"{value:.5g}" if np.isfinite(value) else "undefined")
        self.table.blockSignals(False)
        self.axes.clear()
        for label, value, color in (("Body", body, "#64748b"), ("Features", np.sum(metrics["applied"]), "#0e7490"), ("Total", metrics["total"], "#e87924")):
            self.axes.plot([0, value.real], [0, value.imag], marker="o", label=label, color=color)
        self.axes.set(xlabel="Real F (m)", ylabel="Imaginary F (m)", aspect="equal", adjustable="datalim")
        self.axes.grid(alpha=.25)
        self.axes.legend(loc="upper right")
        self._style_plot()
        self.canvas.draw_idle()
        sigma = metrics["sigma_total"]
        db = 10*np.log10(sigma) if sigma > 0 else -np.inf
        _, frequency, azimuth, elevation = self._result["key"]
        self.total.setText(f"Evaluated sample: {frequency:g} GHz, az {azimuth:g}, el {elevation:g} deg, {self.polarization.currentText()}. Preview total: {sigma:.6g} m2 ({db:.4g} dBsm). Phase and gain sensitivity are undefined at zero gain. Contributions are cached within 16 MiB; choose a new sample and press Inspect to evaluate it.")

    def _stopped(self):
        self._thread.deleteLater()
        self._thread = None
        self.evaluate.setEnabled(True)
        self.family_study.setEnabled(True)

    def select_instance(self, kind, instance_id):
        if self._result is None:
            return
        label = str(kind).title()+" "+str(instance_id)
        if label in self._result["labels"]:
            self.table.selectRow(self._result["labels"].index(label))

    def can_close(self):
        if self._thread is not None:
            self._cancel.set()
            return False
        return True
