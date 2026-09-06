"""Shared Qt variable bindings, widget factories, and native dialogs."""
from __future__ import annotations

import os
from typing import Callable
try:
    from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, Signal
    from PySide6.QtGui import QAction, QColor, QFont, QPainter, QPen
    from PySide6.QtWidgets import (
        QApplication,
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QStackedWidget,
        QSplitter,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )

    QT_AVAILABLE = True
except Exception:  # pragma: no cover - lets the module import without a GUI toolkit
    QT_AVAILABLE = False
    # Minimal fallbacks so module-level class definitions still import; main() raises
    # a friendly error and any GUI use fails loudly at call time.
    QObject = QWidget = QMainWindow = QDialog = object  # type: ignore[assignment,misc]

    def Signal(*_args: object, **_kwargs: object) -> None:  # type: ignore[misc]
        return None


class StringVar(QObject):
    """Lightweight ``tk.StringVar`` work-alike backed by a Qt signal."""

    valueChanged = Signal(str)

    def __init__(self, value: object = "") -> None:
        super().__init__()
        self._value = str(value)

    def get(self) -> str:
        return self._value

    def set(self, value: object) -> None:
        new = str(value)
        if new == self._value:
            return  # idempotent guard breaks bidirectional signal recursion
        self._value = new
        if self.valueChanged is not None:
            self.valueChanged.emit(new)

class BooleanVar(QObject):
    """Lightweight ``tk.BooleanVar`` work-alike backed by a Qt signal."""

    valueChanged = Signal(bool)

    def __init__(self, value: object = False) -> None:
        super().__init__()
        self._value = bool(value)

    def get(self) -> bool:
        return self._value

    def set(self, value: object) -> None:
        new = bool(value)
        if new == self._value:
            return
        self._value = new
        if self.valueChanged is not None:
            self.valueChanged.emit(new)

def bind_line_edit(var: StringVar, edit: QLineEdit) -> None:
    """Two-way bind a StringVar to a QLineEdit (with re-entrancy guard)."""
    edit.setText(var.get())
    guard = {"on": False}

    def from_widget(text: str) -> None:
        if guard["on"]:
            return
        guard["on"] = True
        var.set(text)
        guard["on"] = False

    def from_var(value: str) -> None:
        if guard["on"]:
            return
        guard["on"] = True
        if edit.text() != value:
            edit.setText(value)
        guard["on"] = False

    edit.textChanged.connect(from_widget)
    var.valueChanged.connect(from_var)

def bind_check_box(var: BooleanVar, box: QCheckBox) -> None:
    """Two-way bind a BooleanVar to a QCheckBox."""
    box.setChecked(var.get())
    guard = {"on": False}

    def from_widget(_state: object) -> None:
        if guard["on"]:
            return
        guard["on"] = True
        var.set(box.isChecked())
        guard["on"] = False

    def from_var(value: bool) -> None:
        if guard["on"]:
            return
        guard["on"] = True
        if box.isChecked() != value:
            box.setChecked(value)
        guard["on"] = False

    box.toggled.connect(from_widget)
    var.valueChanged.connect(from_var)

def bind_combo_box(var: StringVar, combo: QComboBox) -> None:
    """Two-way bind a StringVar to a (populated) QComboBox."""
    idx = combo.findText(var.get())
    if idx >= 0:
        combo.setCurrentIndex(idx)
    elif combo.isEditable():
        combo.setCurrentText(var.get())
    guard = {"on": False}

    def from_widget(text: str) -> None:
        if guard["on"]:
            return
        guard["on"] = True
        var.set(text)
        guard["on"] = False

    def from_var(value: str) -> None:
        if guard["on"]:
            return
        guard["on"] = True
        if combo.currentText() != value:
            j = combo.findText(value)
            if j >= 0:
                combo.setCurrentIndex(j)
            elif combo.isEditable():
                combo.setCurrentText(value)
        guard["on"] = False

    combo.currentTextChanged.connect(from_widget)
    var.valueChanged.connect(from_var)

def make_combo(
    values: object,
    var: StringVar,
    *,
    width: int | None = None,
    on_change: Callable[[], None] | None = None,
) -> QComboBox:
    """Build a non-editable QComboBox bound to ``var``."""
    combo = QComboBox()
    combo.addItems([str(v) for v in values])
    bind_combo_box(var, combo)
    if width is not None:
        combo.setMinimumWidth(width)
    if on_change is not None:
        combo.currentTextChanged.connect(lambda _t: on_change())
    return combo

def _qt_filter(filetypes: object) -> str:
    """Convert a tkinter ``filetypes`` list into a Qt name filter string."""
    if not filetypes:
        return ""
    parts = []
    for name, patterns in filetypes:  # type: ignore[misc]
        pats = " ".join(
            "*" if tok in ("*.*", "*", "") else tok
            for tok in str(patterns).replace(",", " ").replace(";", " ").split()
        )
        parts.append(f"{name} ({pats or '*'})")
    return ";;".join(parts)

class _MessageBox:
    """tkinter ``messagebox`` work-alike backed by QMessageBox."""

    @staticmethod
    def showinfo(title: str = "", message: str = "", **kw: object) -> None:
        QMessageBox.information(kw.get("parent"), str(title), str(message))

    @staticmethod
    def showwarning(title: str = "", message: str = "", **kw: object) -> None:
        QMessageBox.warning(kw.get("parent"), str(title), str(message))

    @staticmethod
    def showerror(title: str = "", message: str = "", **kw: object) -> None:
        QMessageBox.critical(kw.get("parent"), str(title), str(message))

    @staticmethod
    def askyesno(title: str = "", message: str = "", **kw: object) -> bool:
        buttons = getattr(QMessageBox, "StandardButton", QMessageBox)
        answer = QMessageBox.question(
            kw.get("parent"),
            str(title),
            str(message),
            buttons.Yes | buttons.No,
            buttons.No,
        )
        return answer == buttons.Yes

class _FileDialog:
    """tkinter ``filedialog`` work-alike backed by QFileDialog."""

    @staticmethod
    def askopenfilename(**kw: object) -> str:
        path, _ = QFileDialog.getOpenFileName(
            kw.get("parent"), str(kw.get("title", "")), "", _qt_filter(kw.get("filetypes"))
        )
        return path

    @staticmethod
    def asksaveasfilename(**kw: object) -> str:
        path, _ = QFileDialog.getSaveFileName(
            kw.get("parent"), str(kw.get("title", "")), "", _qt_filter(kw.get("filetypes"))
        )
        ext = str(kw.get("defaultextension", ""))
        if path and ext and not os.path.splitext(path)[1]:
            path = path + ext
        return path

    @staticmethod
    def askdirectory(**kw: object) -> str:
        return QFileDialog.getExistingDirectory(
            kw.get("parent"), str(kw.get("title", "")), str(kw.get("initialdir", ""))
        )

messagebox = _MessageBox()

filedialog = _FileDialog()
