"""Compact controls for a two-dataset Delta Map."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QToolButton

from GRIM_Backend.plotting.modes import common
from GRIM_Backend.plotting.modes.delta_map_mode import AXIS_ATTRIBUTES


class DeltaMapControls(QFrame):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("deltaMapControls")
        self.reversed = False
        self._fixed_key = None
        self.x_axis, self.y_axis = QComboBox(), QComboBox()
        for combo in (self.x_axis, self.y_axis):
            for name, axis in (("Azimuth", "azimuth"), ("Elevation", "elevation"), ("Frequency", "frequency")):
                combo.addItem(name, axis)
        self.y_axis.setCurrentIndex(2)
        self.fixed_label = QLabel("Fixed elevation")
        self.fixed_value = QComboBox()
        self.fixed_value.setToolTip("Choose a fixed coordinate; horizontal and vertical ranges use the sidebar selections.")
        self.auto_limit = QCheckBox("Auto color limits")
        self.auto_limit.setChecked(True)
        self.limit = QDoubleSpinBox()
        self.limit.setRange(0.001, 1000000)
        self.limit.setDecimals(3)
        self.limit.setValue(5)
        self.limit.setPrefix("± ")
        self.limit.setSuffix(" dB")
        self.limit.setKeyboardTracking(False)
        self.limit.setEnabled(False)
        self.show_values = QCheckBox("Cell values")
        self.show_values.setToolTip("Show signed values when there are at most 200 cells.")
        self.swap = QToolButton(text="Swap A / B")
        self.source_label = QLabel()
        self.source_label.setTextFormat(Qt.PlainText)
        self.source_label.setWordWrap(True)
        self.source_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for col, (label, field) in enumerate(((QLabel("Horizontal"), self.x_axis), (QLabel("Vertical"), self.y_axis), (self.fixed_label, self.fixed_value))):
            layout.addWidget(label, 0, col)
            layout.addWidget(field, 1, col)
        row = QHBoxLayout()
        for widget in (self.auto_limit, self.limit, self.show_values, self.swap):
            row.addWidget(widget)
        row.addStretch(1)
        layout.addLayout(row, 2, 0, 1, 3)
        layout.addWidget(self.source_label, 3, 0, 1, 3)
        self.x_axis.currentIndexChanged.connect(lambda: self._axes_changed(self.x_axis, self.y_axis))
        self.y_axis.currentIndexChanged.connect(lambda: self._axes_changed(self.y_axis, self.x_axis))
        self.fixed_value.currentIndexChanged.connect(self.changed)
        self.auto_limit.toggled.connect(self._auto_changed)
        self.limit.valueChanged.connect(self.changed)
        self.show_values.toggled.connect(self.changed)
        self.swap.clicked.connect(self._swap)
        self.hide()

    @property
    def fixed_axis(self):
        return next(axis for axis in ("azimuth", "elevation", "frequency")
                    if axis not in (self.x_axis.currentData(), self.y_axis.currentData()))

    def _axes_changed(self, edited, other):
        if edited.currentData() == other.currentData():
            other.blockSignals(True)
            other.setCurrentIndex(next(i for i in range(other.count()) if other.itemData(i) != edited.currentData()))
            other.blockSignals(False)
        self.changed.emit()

    def _auto_changed(self, checked):
        self.limit.setEnabled(not checked)
        self.changed.emit()

    def _swap(self):
        self.reversed = not self.reversed
        self.changed.emit()

    def configure(self, reference, selections, datasets):
        for combo in (self.x_axis, self.y_axis):
            combo.blockSignals(True)
            for i in range(combo.count()):
                axis = combo.itemData(i)
                combo.setItemText(i, "Frequency" if axis == "frequency" else common.angular_axis_name(reference, axis))
            combo.blockSignals(False)
        axis = self.fixed_axis
        name = "Frequency" if axis == "frequency" else common.angular_axis_name(reference, axis)
        unit = common.axis_unit(reference, axis)
        self.fixed_label.setText(f"Fixed {name.lower()} ({unit})")
        key = (id(reference), axis, tuple(selections[axis]))
        if key != self._fixed_key:
            preferred = self.fixed_value.currentData() if self._fixed_key and self._fixed_key[:2] == key[:2] else None
            if len(selections[axis]) == 1:
                preferred = selections[axis][0]
            values = sorted(set(float(v) for v in getattr(reference, AXIS_ATTRIBUTES[axis])))
            self.fixed_value.blockSignals(True)
            self.fixed_value.clear()
            for value in values:
                self.fixed_value.addItem(f"{value:g}", value)
            index = self.fixed_value.findData(preferred)
            self.fixed_value.setCurrentIndex(index if index >= 0 else 0)
            self.fixed_value.blockSignals(False)
            self._fixed_key = key
        ordered = list(reversed(datasets)) if self.reversed else datasets
        self.source_label.setText(f"A: {ordered[0][0]}   |   B: {ordered[1][0]}   |   A - B (dB)")

    def options(self):
        return {"x_axis": self.x_axis.currentData(), "y_axis": self.y_axis.currentData(),
                "limit": None if self.auto_limit.isChecked() else self.limit.value(),
                "show_values": self.show_values.isChecked()}
