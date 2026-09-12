"""Column-labeling and file-conversion UI shared with the GRIM importer."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QFileDialog,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QSpinBox,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .table_conversion import (
    ColumnMapping, DELIMITERS, TableOptions, UNITS, converted_rows, export_table,
    preview_table, suggest_column, validate_mapping,
)


class _ConversionThread(QThread):
    def __init__(self, operation, parent):
        super().__init__(parent)
        self.operation = operation
        self.result = None
        self.error = ""

    def run(self):
        try:
            self.result = self.operation()
        except Exception as exc:
            self.error = str(exc)


class FileConverterDialog(QDialog):
    """Preview a source, collect mappings, then run conversion off the UI thread."""

    def __init__(self, parent=None, path="", *, importing=False):
        super().__init__(parent)
        self.importing = importing
        self.setWindowTitle("Label columns for GRIM" if importing else "File Converter")
        self.resize(1080, 730)
        self.setAcceptDrops(not importing)
        self.job = None
        self.names = ()
        self.sample = []
        self.preview_options = None
        self.preview_path = ""
        layout = QVBoxLayout(self)
        self.intro = QLabel(
            "Label the columns and declare their units. Use Constant to fill a missing column for every row. "
            "Uncheck columns you do not need. Input files are kept unchanged."
        )
        self.intro.setWordWrap(True)
        layout.addWidget(self.intro)
        self.controls = QWidget()
        controls_layout = QVBoxLayout(self.controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        source_row = QHBoxLayout()
        self.path_edit = QLineEdit(str(path))
        self.path_edit.setReadOnly(importing)
        self.path_edit.setPlaceholderText("Open or drop a CSV / ASCII text table")
        source_row.addWidget(self.path_edit, 1)
        self.open_button = QPushButton("Open file…")
        self.open_button.setVisible(not importing)
        self.open_button.clicked.connect(self.open_file)
        source_row.addWidget(self.open_button)
        controls_layout.addLayout(source_row)
        options_row = QHBoxLayout()
        self.delimiter = QComboBox()
        self.delimiter.addItems(DELIMITERS)
        self.header = QComboBox()
        self.header.addItems(("Auto", "Yes", "No"))
        self.skip_rows = QSpinBox()
        self.skip_rows.setRange(0, 1000000)
        self.encoding = QComboBox()
        self.encoding.addItems(("utf-8-sig", "cp1252"))
        for label, widget in (("Delimiter", self.delimiter), ("Header", self.header),
                              ("Skip lines", self.skip_rows), ("Encoding", self.encoding)):
            options_row.addWidget(QLabel(label))
            options_row.addWidget(widget)
        self.refresh_button = QPushButton("Refresh preview")
        self.refresh_button.setToolTip("Reread the source and reset the column mapping.")
        self.refresh_button.clicked.connect(self.refresh_preview)
        options_row.addWidget(self.refresh_button)
        controls_layout.addLayout(options_row)
        self.preview_label = QLabel("Source preview · first 20 data rows")
        controls_layout.addWidget(self.preview_label)
        self.preview = QTableWidget()
        self.preview.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.preview.horizontalHeader().setMinimumSectionSize(140)
        self.preview.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.preview.setMinimumHeight(110)
        controls_layout.addWidget(self.preview, 1)
        controls_layout.addWidget(QLabel("Output columns · select a source column or a constant"))
        self.mapping = QTableWidget(0, 5)
        self.mapping.setHorizontalHeaderLabels(("Use / Variable", "Source column", "Constant value",
                                                "Input unit", "Output unit"))
        self.mapping.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.mapping.setMinimumHeight(180)
        controls_layout.addWidget(self.mapping, 2)
        self.extra_layout = QHBoxLayout()
        self.add_button = QPushButton("Add constant column")
        self.add_button.setVisible(not importing)
        self.add_button.clicked.connect(lambda: self.add_mapping(f"value_{self.mapping.rowCount() + 1}"))
        self.extra_layout.addWidget(self.add_button)
        self.preview_conversion_button = QPushButton("Preview conversion")
        self.preview_conversion_button.clicked.connect(self.preview_conversion)
        self.extra_layout.addWidget(self.preview_conversion_button)
        self.extra_layout.addStretch()
        self.output_delimiter = QComboBox()
        self.output_delimiter.addItems(("Comma", "Tab", "Whitespace", "Semicolon"))
        if not importing:
            self.extra_layout.addWidget(QLabel("Output delimiter"))
            self.extra_layout.addWidget(self.output_delimiter)
        controls_layout.addLayout(self.extra_layout)
        layout.addWidget(self.controls, 1)
        self.status = QLabel("Choose a file to begin.")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.PlainText)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status)
        footer = QHBoxLayout()
        footer.addStretch()
        self.cancel_button = QPushButton("Cancel" if importing else "Close")
        self.cancel_button.clicked.connect(self.reject)
        self.submit_button = QPushButton("Import into GRIM" if importing else "Convert and save…")
        self.submit_button.setEnabled(False)
        self.submit_button.clicked.connect(self.submit)
        footer.addWidget(self.cancel_button)
        footer.addWidget(self.submit_button)
        layout.addLayout(footer)
        if path:
            self.refresh_preview()

    def options(self):
        return TableOptions(self.delimiter.currentText(), self.header.currentText(),
                            self.skip_rows.value(), self.encoding.currentText())

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select a delimited table", "",
            "Text tables (*.csv *.txt *.dat *.asc *.ascii *.tsv);;All files (*)")
        if path:
            self.path_edit.setText(path)
            self.refresh_preview()

    def dragEnterEvent(self, event):
        if not self.importing and self.job is None and event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if not self.importing and self.job is None and len(paths) == 1:
            self.path_edit.setText(paths[0])
            self.refresh_preview()
            event.acceptProposedAction()

    def refresh_preview(self):
        self.submit_button.setEnabled(False)
        self.names = ()
        self.mapping.setRowCount(0)
        self.preview.setRowCount(0)
        try:
            path, options = self.path_edit.text().strip(), self.options()
            self.names, self.sample = preview_table(path, options)
            self.preview_path, self.preview_options = path, options
            self.preview_label.setText("Source preview · first 20 data rows")
            self.preview.setColumnCount(len(self.names))
            self.preview.setHorizontalHeaderLabels([f"{i + 1}: {name}" for i, name in enumerate(self.names)])
            self.preview.setRowCount(len(self.sample))
            self.preview.setVerticalHeaderLabels([str(line) for line, _ in self.sample])
            for row_index, (_, row) in enumerate(self.sample):
                for col, value in enumerate(row):
                    self.preview.setItem(row_index, col, QTableWidgetItem(value))
            self.configure_mapping()
            self.status.setText("Review labels and units, then " + ("import." if self.importing else
                "save a converted copy. Output headers include the selected units."))
            self.submit_button.setEnabled(True)
        except Exception as exc:
            self.status.setText(str(exc))

    def configure_mapping(self):
        for index, label in enumerate(self.names):
            name, unit = suggest_column(label)
            self.add_mapping(name, source=index, input_unit=unit, output_unit=unit)

    def preview_conversion(self):
        records = None
        try:
            path, options, columns, names = self.snapshot()
            records = converted_rows(path, options, columns, names)
            sample = []
            for line_no, row in records:
                sample.append((line_no, row))
                if len(sample) >= 20:
                    break
            self.preview.setColumnCount(len(columns))
            self.preview.setHorizontalHeaderLabels([
                f"{c.name} ({c.output_unit})" if c.output_unit != "As written" else c.name for c in columns])
            self.preview.setRowCount(len(sample))
            self.preview.setVerticalHeaderLabels([str(line) for line, _ in sample])
            for row_index, (_, row) in enumerate(sample):
                for col, column in enumerate(columns):
                    value = row[column.name]
                    self.preview.setItem(row_index, col, QTableWidgetItem(
                        format(value, ".12g") if isinstance(value, float) else str(value)))
            self.preview_label.setText("Converted preview · first 20 data rows (refresh to see the source)")
            self.status.setText("Preview updated. All rows will be validated when you import or save.")
        except Exception as exc:
            self.status.setText(str(exc))
        finally:
            if records is not None:
                records.close()

    def add_mapping(self, name, *, source=None, constant="", input_unit="As written",
                    output_unit="As written", checked=True, locked=False,
                    input_units=None, output_units=None):
        row = self.mapping.rowCount()
        self.mapping.insertRow(row)
        item = QTableWidgetItem(name)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        if locked:
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        self.mapping.setItem(row, 0, item)
        selector = QComboBox()
        selector.addItem("Choose column…", -2)
        selector.addItem("Constant (all rows)", None)
        for index, label in enumerate(self.names):
            selector.addItem(f"{index + 1}: {label}", index)
        selector.setCurrentIndex(1 if source is None else selector.findData(source))
        self.mapping.setCellWidget(row, 1, selector)
        value = QLineEdit(str(constant))
        value.setEnabled(source is None)
        selector.currentIndexChanged.connect(lambda _index: value.setEnabled(selector.currentData() is None))
        self.mapping.setCellWidget(row, 2, value)
        for col, units, selected in ((3, input_units, input_unit), (4, output_units, output_unit)):
            combo = QComboBox()
            combo.addItems(units or UNITS)
            if combo.findText(selected) < 0:
                combo.insertItem(0, selected)
            combo.setCurrentText(selected)
            if units and len(units) == 1:
                combo.setEnabled(False)
            self.mapping.setCellWidget(row, col, combo)

    def snapshot(self):
        if self.options() != self.preview_options or self.path_edit.text().strip() != self.preview_path:
            raise ValueError("Refresh the preview after changing the source or parsing options.")
        columns = []
        for row in range(self.mapping.rowCount()):
            item = self.mapping.item(row, 0)
            if item.checkState() != Qt.Checked:
                continue
            columns.append(ColumnMapping(item.text().strip(), self.mapping.cellWidget(row, 1).currentData(),
                self.mapping.cellWidget(row, 2).text(), self.mapping.cellWidget(row, 3).currentText(),
                self.mapping.cellWidget(row, 4).currentText()))
        validate_mapping(columns, len(self.names))
        return self.preview_path, self.preview_options, tuple(columns), tuple(self.names)

    def submit(self):
        try:
            path, options, columns, names = self.snapshot()
            separator = {"Comma": ",", "Tab": "\t", "Whitespace": " ", "Semicolon": ";"}[
                self.output_delimiter.currentText()]
            suffix = ".csv" if separator in (",", ";") else ".txt"
            proposed = str(Path(path).with_name(Path(path).stem + "_converted" + suffix))
            destination, _ = QFileDialog.getSaveFileName(self, "Save converted table", proposed,
                                                       "CSV (*.csv);;Text table (*.txt);;All files (*)")
            if not destination:
                return
            self.start_job(lambda: (export_table(path, destination, options, columns,
                delimiter=separator, expected_names=names), destination))
        except Exception as exc:
            self.status.setText(str(exc))

    def start_job(self, operation):
        if self.job is not None:
            return
        self.controls.setEnabled(False)
        self.submit_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.status.setText("Validating and converting all rows…")
        self.job = _ConversionThread(operation, self)
        self.job.finished.connect(self.job_finished)
        self.job.start()

    def job_finished(self):
        job = self.job
        self.job = None
        self.controls.setEnabled(True)
        self.submit_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        if job.error:
            self.status.setText(job.error)
        else:
            self.conversion_finished(job.result)
        job.deleteLater()

    def conversion_finished(self, result):
        self.status.setText(f"Saved {result[0]:,} rows to {result[1]}")

    def done(self, result):
        if self.job is None:
            super().done(result)

    def closeEvent(self, event):
        if self.job is not None:
            event.ignore()
        else:
            super().closeEvent(event)
