"""Generic Qt desktop shell (PySide6, or PySide2 on Python 3.6)."""

import traceback
from typing import Any

try:
    from PySide6.QtCore import (
        QObject, QRunnable, QSettings, Qt, QThreadPool, Signal
    )
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout,
        QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox,
        QPushButton, QPlainTextEdit, QSplitter, QVBoxLayout, QWidget,
    )
except ImportError:  # PySide6 does not provide Python 3.6 wheels.
    from PySide2.QtCore import (  # type: ignore
        QObject, QRunnable, QSettings, Qt, QThreadPool, Signal
    )
    from PySide2.QtWidgets import (  # type: ignore
        QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout,
        QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox,
        QPushButton, QPlainTextEdit, QSplitter, QVBoxLayout, QWidget,
    )

from .registry import CHECK, CHOICE, DIR, FieldSpec, ToolRegistry, ToolSpec, default_registry


STYLE_SHEET = """
QMainWindow, QWidget {
    background: #f4f7fb;
    color: #172033;
    font-size: 13px;
}
QListWidget {
    background: #172033;
    color: #dbe5f5;
    border: 0;
    padding: 10px 6px;
    outline: 0;
}
QListWidget::item {
    border-radius: 7px;
    margin: 3px;
    padding: 11px 12px;
}
QListWidget::item:hover { background: #263653; }
QListWidget::item:selected {
    background: #2878d0;
    color: white;
}
QLineEdit, QComboBox, QPlainTextEdit {
    background: white;
    border: 1px solid #cbd5e3;
    border-radius: 6px;
    padding: 7px;
    selection-background-color: #2878d0;
}
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus {
    border: 1px solid #2878d0;
}
QPushButton {
    background: #e7edf5;
    border: 0;
    border-radius: 6px;
    padding: 8px 14px;
}
QPushButton:hover { background: #d9e4f2; }
QPushButton#runButton {
    background: #2878d0;
    color: white;
    font-weight: 600;
    padding: 10px 24px;
}
QPushButton#runButton:hover { background: #1f68b7; }
QPushButton#runButton:disabled { background: #9db7d3; }
QLabel#titleLabel {
    color: #172033;
    font-size: 24px;
    font-weight: 700;
}
QLabel#descriptionLabel { color: #526079; }
QLabel#sectionLabel {
    color: #526079;
    font-size: 11px;
    font-weight: 700;
}
QSplitter::handle { background: #d9e1ec; width: 1px; }
"""


class WorkerSignals(QObject):
    succeeded = Signal(object)
    failed = Signal(str)


class ToolWorker(QRunnable):
    def __init__(self, spec: 'ToolSpec', values: 'dict[str, Any]') -> 'None':
        super().__init__()
        self.spec = spec
        self.values = values
        self.signals = WorkerSignals()

    def run(self) -> 'None':
        try:
            self.signals.succeeded.emit(self.spec.function(**self.values))
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


class DirectoryField(QWidget):
    def __init__(self, parent: 'QWidget | None' = None) -> 'None':
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        layout.addWidget(self.edit, 1)
        layout.addWidget(browse)

    def _browse(self) -> 'None':
        selected = QFileDialog.getExistingDirectory(self, "Select folder", self.edit.text())
        if selected:
            self.edit.setText(selected)

    def value(self) -> 'str':
        return self.edit.text().strip()

    def set_value(self, value: 'str') -> 'None':
        self.edit.setText(value)


class MainWindow(QMainWindow):
    def __init__(self, registry: 'ToolRegistry | None' = None) -> 'None':
        super().__init__()
        self.registry = registry or default_registry()
        self.specs = self.registry.tools()
        self.widgets: 'dict[str, QWidget]' = {}
        self.active_spec: 'ToolSpec | None' = None
        self.settings = QSettings("CEM Tools", "CEM Tools")
        self.pool = QThreadPool.globalInstance()
        self.setWindowTitle("CEM Tools")
        self.resize(940, 640)
        self.setStyleSheet(STYLE_SHEET)

        self.tool_list = QListWidget()
        self.tool_list.addItems([spec.title for spec in self.specs])
        self.title = QLabel()
        self.title.setObjectName("titleLabel")
        self.description = QLabel()
        self.description.setObjectName("descriptionLabel")
        self.description.setWordWrap(True)
        self.form_host = QWidget()
        self.form = QFormLayout(self.form_host)
        self.form.setContentsMargins(0, 18, 0, 10)
        self.form.setHorizontalSpacing(24)
        self.form.setVerticalSpacing(12)
        self.run_button = QPushButton("Run tool")
        self.run_button.setObjectName("runButton")
        self.run_button.clicked.connect(self._run)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(100)
        self.log.setPlaceholderText("Run summaries and errors appear here.")
        activity_label = QLabel("ACTIVITY")
        activity_label.setObjectName("sectionLabel")

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(30, 26, 30, 26)
        right_layout.setSpacing(10)
        right_layout.addWidget(self.title)
        right_layout.addWidget(self.description)
        right_layout.addWidget(self.form_host)
        right_layout.addWidget(self.run_button, 0, Qt.AlignmentFlag.AlignRight)
        right_layout.addSpacing(8)
        right_layout.addWidget(activity_label)
        right_layout.addWidget(self.log, 1)
        splitter = QSplitter()
        splitter.addWidget(self.tool_list)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([245, 695])
        self.setCentralWidget(splitter)
        self.tool_list.currentRowChanged.connect(self._select_tool)
        if self.specs:
            self.tool_list.setCurrentRow(0)

    def _clear_form(self) -> 'None':
        while self.form.rowCount():
            self.form.removeRow(0)
        self.widgets.clear()

    def _make_widget(self, field: 'FieldSpec') -> 'QWidget':
        if field.kind == DIR:
            widget = DirectoryField()
            widget.set_value(str(self.settings.value(f"{self.active_spec.identifier}/{field.name}", "")))
            return widget
        if field.kind == CHECK:
            widget = QCheckBox()
            widget.setChecked(bool(field.default))
            return widget
        if field.kind == CHOICE:
            widget = QComboBox()
            widget.addItems(field.options)
            if field.default in field.options:
                widget.setCurrentText(field.default)
            return widget
        widget = QLineEdit("" if field.default is None else str(field.default))
        return widget

    def _select_tool(self, index: 'int') -> 'None':
        if index < 0:
            return
        self.active_spec = self.specs[index]
        self._clear_form()
        self.title.setText(self.active_spec.title)
        self.description.setText(self.active_spec.description)
        for field in self.active_spec.fields:
            widget = self._make_widget(field)
            self.widgets[field.name] = widget
            self.form.addRow(field.label + ("" if not field.required else " *"), widget)
        if self.active_spec.identifier == "rename":
            checkbox = self.widgets["in_place"]
            checkbox.toggled.connect(self.widgets["output_dir"].setDisabled)
        self.log.clear()

    @staticmethod
    def _widget_value(widget: 'QWidget') -> 'Any':
        if isinstance(widget, DirectoryField):
            return widget.value()
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QComboBox):
            return widget.currentText()
        return widget.text().strip()

    def _run(self) -> 'None':
        if self.active_spec is None:
            return
        values: 'dict[str, Any]' = {}
        missing = []
        for field in self.active_spec.fields:
            value = self._widget_value(self.widgets[field.name])
            if field.required and not value:
                missing.append(field.label)
            values[field.name] = value
            if field.kind == DIR and value:
                self.settings.setValue(f"{self.active_spec.identifier}/{field.name}", value)
        if missing:
            QMessageBox.warning(self, "Missing input", "Required: " + ", ".join(missing))
            return
        if (
            self.active_spec.identifier == "rename"
            and not values["in_place"]
            and not values["output_dir"]
        ):
            QMessageBox.warning(
                self, "Missing input",
                "Choose an output folder or select Rename in place.",
            )
            return
        if self.active_spec.identifier == "rename" and values["in_place"]:
            values["output_dir"] = None
        self.run_button.setEnabled(False)
        self.log.appendPlainText(f"Running {self.active_spec.title}…")
        worker = ToolWorker(self.active_spec, values)
        worker.signals.succeeded.connect(self._succeeded)
        worker.signals.failed.connect(self._failed)
        self.pool.start(worker)

    def _succeeded(self, result: 'Any') -> 'None':
        self.run_button.setEnabled(True)
        self.log.appendPlainText(result.summary())
        for warning in result.warnings:
            self.log.appendPlainText("Warning: " + warning)

    def _failed(self, details: 'str') -> 'None':
        self.run_button.setEnabled(True)
        final_line = details.strip().splitlines()[-1]
        self.log.appendPlainText(details)
        QMessageBox.critical(self, "Tool failed", final_line)


def run_gui() -> 'int':
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    execute = getattr(application, "exec", None)
    return execute() if execute is not None else application.exec_()
