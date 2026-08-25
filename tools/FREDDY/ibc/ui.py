from __future__ import annotations

import math
import os
import queue
import random
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except Exception:
    np = None  # type: ignore[assignment]
    NUMPY_AVAILABLE = False

try:
    import scipy.optimize as _scipy_optimize

    SCIPY_AVAILABLE = True
except Exception:
    _scipy_optimize = None  # type: ignore[assignment]
    SCIPY_AVAILABLE = False

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

if QT_AVAILABLE:
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        MPL_AVAILABLE = True
    except Exception:
        MPL_AVAILABLE = False
else:
    MPL_AVAILABLE = False


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

from .batch import (
    IbcBatchItem,
    export_pec_ibc_thickness_batch,
    plan_ibc_thickness_batch,
)
from .compute import (
    INCH_TO_M,
    MIX_RULE_LABELS,
    MIX_RULE_DESCRIPTIONS,
    MIX_RULES,
    InverseCandidate,
    LayerConfig,
    LoadedLayer,
    MaterialTable,
    MixCandidate,
    MixComponent,
    UncertaintyConfig,
    align_phase_degrees,
    blend_density_gcc,
    build_uncertainty_scales,
    combine_mix,
    compute_angle_metrics,
    compute_angle_metrics_many,
    compute_stack_impedance_many,
    interp_complex_many,
    interp_components_on_grid,
    is_nominal_scale,
    make_frequency_sweep,
    make_sweep,
    mix_material_tables,
    mix_model_advisories,
    normalize_backing,
    normalize_mix_rule,
    normalize_wave_polarization,
    parts_to_fractions,
    prepare_layer_properties_many,
    prepare_layer_wave_terms_many,
    property_match_error,
    property_match_error_curve,
    project_bounded_fractions,
    snap_to_increment,
    validate_incidence_angle,
    validate_fraction_bounds,
    validate_sweep_coverage,
    weight_fractions_from_volume,
)
from .io import (
    HZ_PER_GHZ,
    _atomic_text_file,
    layer_config_from_dict,
    layer_config_to_dict,
    load_project_file,
    read_material_table,
    save_project_file,
    uncertainty_report_path,
    write_impedance_bundle,
    write_material_table,
)
from .plot import nearest_index, style_axis, style_colorbar

APP_ACRONYM = "FREDDY"
APP_NAME = "Frequency-Dependent Reflection and EM Dielectric Dimensional Yield"
APP_TITLE = f"{APP_ACRONYM} - {APP_NAME}"
ABOUT_TEXT = (
    f"{APP_TITLE}\n\n"
    f"Acronym: {APP_NAME}\n\n"
    "Physical scope:\n"
    "Infinite planar material-stack reflection, transmission, absorption, and input impedance.\n"
    "FREDDY does not calculate finite-object RCS or dBsm.\n\n"
    "Angle convention:\n"
    "0 deg = normal incidence (broadside)\n"
    "angles must be < 90 deg; exact grazing has singular field normalization\n\n"
    "Polarization convention:\n"
    "Use plane-wave TE/TM labels. Legacy HH=TE and VV=TM aliases follow the\n"
    "usual vertical plane-of-incidence convention, but the companion 2D RCS\n"
    "elevation-cut solver uses different HH/VV aliases.\n\n"
    "Loss metric definitions:\n"
    "loss_db = 20*log10(|x|)\n"
    "metal_loss_db uses x = Gamma_metal\n"
    "air_loss_db uses x = Gamma_air\n"
    "insertion_loss_db uses x = S21\n\n"
    "Sign interpretation:\n"
    "negative loss_db => |x| < 1 (attenuation)\n"
    "zero loss_db => |x| = 1\n"
    "positive loss_db => |x| > 1 (effective gain/non-passive)\n\n"
    "Absorption metric definitions:\n"
    "metal_absorption_db = 10*log10(1 - |Gamma_metal|^2)\n"
    "air_absorption_db = 10*log10(1 - |Gamma_air|^2 - |S21|^2)\n\n"
    "Metal absorption: power absorbed by the stack on a\n"
    "PEC ground plane in dB (0 dB = perfect absorption).\n"
    "Air absorption: power absorbed by a free-standing\n"
    "slab in dB (accounts for both reflection and transmission)."
)

# --- GRIM blue/slate palette --------------------------------------------
# The dark theme shares GRIM's exact application chrome colors. The light
# variant keeps the same blue identity for standalone users who prefer a
# bright workspace. Plot traces use blue, amber, and violet so adjacent data
# remains distinguishable without the former red/green theme.

LIGHT_THEME = {
    "window_bg": "#f1f5f9",
    "panel_bg": "#e2e8f0",
    "head_bg": "#dbeafe",
    "text": "#0f172a",
    "muted_text": "#475569",
    "field_bg": "#f8fafc",
    "field_fg": "#0f172a",
    "field_disabled_bg": "#cbd5e1",
    "field_disabled_fg": "#64748b",
    "button_bg": "#e2e8f0",
    "button_active_bg": "#bfdbfe",
    "selection_bg": "#2563eb",
    "selection_fg": "#ffffff",
    "accent": "#1d4ed8",
    "preview_bg": "#f8fafc",
    "preview_border": "#3b82f6",
    "preview_outline": "#64748b",
    "preview_text": "#0f172a",
    "preview_empty": "#64748b",
    "preview_layer_text": "#eff6ff",
    "preview_layer_border": "#f8fafc",
    "layer_colors": [
        "#1d4ed8",
        "#0369a1",
        "#4f46e5",
        "#0e7490",
        "#2563eb",
        "#475569",
        "#7c3aed",
        "#0284c7",
    ],
    "plot_bg": "#f8fafc",
    "plot_axes_bg": "#ffffff",
    "plot_text": "#0f172a",
    "plot_spine": "#64748b",
    "plot_grid": "#cbd5e1",
    "plot_line_freq": "#0369a1",
    "plot_line_angle": "#6d28d9",
    "plot_worst": "#b45309",
    "plot_crosshair": "#0f172a",
}

DARK_THEME = {
    "window_bg": "#0f172a",
    "panel_bg": "#0b1222",
    "head_bg": "#172554",
    "text": "#dbeafe",
    "muted_text": "#94a3b8",
    "field_bg": "#0b1222",
    "field_fg": "#dbeafe",
    "field_disabled_bg": "#172554",
    "field_disabled_fg": "#64748b",
    "button_bg": "#0b1222",
    "button_active_bg": "#1d4ed8",
    "selection_bg": "#2563eb",
    "selection_fg": "#ffffff",
    "accent": "#3b82f6",
    "preview_bg": "#0b1222",
    "preview_border": "#1e3a8a",
    "preview_outline": "#64748b",
    "preview_text": "#dbeafe",
    "preview_empty": "#94a3b8",
    "preview_layer_text": "#eff6ff",
    "preview_layer_border": "#0b1222",
    "layer_colors": [
        "#1e3a8a",
        "#1d4ed8",
        "#172554",
        "#2563eb",
        "#1e40af",
        "#3b82f6",
        "#334155",
        "#0284c7",
    ],
    "plot_bg": "#0b1222",
    "plot_axes_bg": "#0b1222",
    "plot_text": "#dbeafe",
    "plot_spine": "#1e3a8a",
    "plot_grid": "#475569",
    "plot_line_freq": "#38bdf8",
    "plot_line_angle": "#a78bfa",
    "plot_worst": "#fbbf24",
    "plot_crosshair": "#dbeafe",
}

HEATMAP_METRIC_OPTIONS = [
    ("Metal backed loss (dB)", "metal_loss_db"),
    ("Metal phase (deg)", "metal_phase_deg"),
    ("Metal absorption (dB)", "metal_absorption_db"),
    ("Air backed loss (dB)", "air_loss_db"),
    ("Air phase (deg)", "air_phase_deg"),
    ("Air absorption (dB)", "air_absorption_db"),
    ("Insertion loss (dB)", "insertion_loss_db"),
    ("Insertion phase (deg)", "insertion_phase_deg"),
]
HEATMAP_METRIC_KEYS = [key for _label, key in HEATMAP_METRIC_OPTIONS]
PHASE_METRIC_KEYS = {
    "metal_phase_deg",
    "air_phase_deg",
    "insertion_phase_deg",
}


@dataclass
class HeatmapView:
    """One computed heatmap ready to draw: metric grids indexed [freq][x], the
    optional uncertainty envelopes, and how to label the x axis. The Off Angle
    and Thickness tabs differ only in what x is, so both share the plotting,
    slicing, and image-export paths."""

    results: dict[str, list[list[float]] | list[float]]
    unc_min: dict[str, list[list[float]]] | None
    unc_max: dict[str, list[list[float]]] | None
    x_key: str
    x_name: str
    x_unit: str

    @property
    def x_vals(self) -> list[float]:
        return self.results[self.x_key]  # type: ignore[return-value]

    @property
    def freqs(self) -> list[float]:
        return self.results["freq_ghz"]  # type: ignore[return-value]

    @property
    def x_label(self) -> str:
        return f"{self.x_name} ({self.x_unit})"
UNCERTAINTY_VIEW_OPTIONS = [
    ("Nominal", "nominal"),
    ("Min", "min"),
    ("Max", "max"),
    ("Span (max-min)", "span"),
]
INVERSE_SCORE_MODE_OPTIONS = (
    "Worst-case mean metal loss (robust)",
    "Average mean metal loss (robust)",
)
MIX_RULE_LABEL_OPTIONS = tuple(MIX_RULE_LABELS[key] for key in MIX_RULES)
# Material Mix predicts a single homogeneous effective layer. Its performance
# target is therefore deliberately narrower than the separate multilayer-stack
# inverse-design workflow, and neither workflow calculates finite-object RCS.
MIX_OBJECTIVE_FORWARD = "Predict properties from a known recipe"
MIX_OBJECTIVE_PROPERTY = "Find a recipe for target properties"
MIX_OBJECTIVE_PERFORMANCE = "Find a recipe for target stack performance"
MIX_OBJECTIVE_OPTIONS = (
    MIX_OBJECTIVE_FORWARD,
    MIX_OBJECTIVE_PROPERTY,
    MIX_OBJECTIVE_PERFORMANCE,
)
MIX_PROP_SOURCE_OPTIONS = ("Constant values", "Material file")
MIX_PERFORMANCE_METRIC_OPTIONS = (
    ("PEC-backed reflection |Γ| (dB)", "metal_loss_db", "at_most", "dB", -10.0),
    ("PEC-backed absorption (%)", "metal_absorption_db", "at_least", "%", 90.0),
    ("Air-backed reflection |Γ| (dB)", "air_loss_db", "at_most", "dB", -10.0),
    ("Air-backed absorption (%)", "air_absorption_db", "at_least", "%", 50.0),
    ("Air-backed transmission |S21| (dB)", "insertion_loss_db", "at_most", "dB", -10.0),
)
MIX_PERFORMANCE_SPEC_BY_LABEL = {
    label: {
        "label": label,
        "metric_key": metric_key,
        "direction": direction,
        "unit": unit,
        "default_target": default_target,
    }
    for label, metric_key, direction, unit, default_target in MIX_PERFORMANCE_METRIC_OPTIONS
}
# Corner-aggregation labels for property mismatch or performance gap.
MIX_SCORE_MODE_OPTIONS = (
    "Worst-case across uncertainty corners (robust)",
    "Average across uncertainty corners",
)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILTIN_MATERIAL_PRESETS = {
    # Air is exact and is the only bundled material until measured/validated
    # property data are available. Do not ship generic unvalidated presets.
    "Air (reference)": str(_PROJECT_ROOT / "materials" / "air_reference.csv"),
}
_T = TypeVar("_T")


class CollapsibleFrame(QWidget):
    """A section with a clickable header that shows/hides its body."""

    def __init__(
        self,
        text: str,
        *,
        expanded: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._text = text
        self._expanded = expanded

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._header = QToolButton()
        self._header.setObjectName("CollapsibleHeader")
        self._header.setCursor(Qt.PointingHandCursor)
        self._header.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self._header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._header.clicked.connect(self.toggle)
        outer.addWidget(self._header)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        outer.addWidget(separator)

        self.body = QWidget()
        outer.addWidget(self.body)

        self._refresh_label()
        self.body.setVisible(expanded)

    def _refresh_label(self) -> None:
        arrow = "▾" if self._expanded else "▸"
        self._header.setText(f"{arrow}  {self._text}")

    def toggle(self) -> None:
        if self._expanded:
            self.collapse()
        else:
            self.expand()

    def expand(self) -> None:
        if self._expanded:
            return
        self._expanded = True
        self._refresh_label()
        self.body.setVisible(True)

    def collapse(self) -> None:
        if not self._expanded:
            return
        self._expanded = False
        self._refresh_label()
        self.body.setVisible(False)


class LayerPreview(QWidget):
    """Canvas-like widget that delegates painting to a callback."""

    def __init__(
        self,
        paint_cb: Callable[[QPainter], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._paint_cb = paint_cb
        self.setMinimumSize(250, 250)
        self.setObjectName("LayerPreview")

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        try:
            self._paint_cb(painter)
        finally:
            painter.end()

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        self.update()


def _parse_optional_thickness(text: str, label: str) -> float | None:
    stripped = text.strip()
    if not stripped:
        return None
    value = float(stripped)
    if value <= 0:
        raise ValueError(f"{label} must be > 0.")
    return value


class LayerDialog(QDialog):
    def __init__(
        self,
        parent: QWidget | None = None,
        initial: LayerConfig | None = None,
        presets: dict[str, str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Layer")
        self.setModal(True)
        self.result: LayerConfig | None = None
        self.presets = presets or {}

        init = initial or LayerConfig(
            thickness_in=0.125,
            anisotropic=False,
            file_0deg="material.csv",
            file_90deg="",
            polarization_deg=0.0,
        )

        self.thickness_var = StringVar(str(init.thickness_in))
        self.aniso_var = BooleanVar(init.anisotropic)
        self.file_0deg_var = StringVar(init.file_0deg)
        self.file_90deg_var = StringVar(init.file_90deg)
        self.pol_var = StringVar(str(init.polarization_deg))
        self.preset_var = StringVar("")
        self.inv_t_min_var = StringVar(
            "" if init.inv_t_min_in is None else f"{init.inv_t_min_in:g}"
        )
        self.inv_t_max_var = StringVar(
            "" if init.inv_t_max_in is None else f"{init.inv_t_max_in:g}"
        )
        self.inv_t_acc_var = StringVar(
            "" if init.inv_t_accuracy_in is None else f"{init.inv_t_accuracy_in:g}"
        )

        grid = QGridLayout()
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        grid.addWidget(QLabel("Thickness (in)"), 0, 0, Qt.AlignLeft)
        thickness_edit = QLineEdit()
        bind_line_edit(self.thickness_var, thickness_edit)
        grid.addWidget(thickness_edit, 0, 1)

        grid.addWidget(QLabel("Preset material"), 1, 0, Qt.AlignLeft)
        preset_values = [""] + sorted(self.presets.keys())
        self.preset_combo = make_combo(preset_values, self.preset_var, width=220)
        grid.addWidget(self.preset_combo, 1, 1)
        use_btn = QPushButton("Use")
        use_btn.clicked.connect(self._apply_preset)
        grid.addWidget(use_btn, 1, 2)

        self.aniso_check = QCheckBox(
            "Directional layer (principal-axis 0 deg / 90 deg files)"
        )
        bind_check_box(self.aniso_var, self.aniso_check)
        self.aniso_check.clicked.connect(self._sync_state)
        grid.addWidget(self.aniso_check, 2, 0, 1, 3, Qt.AlignLeft)

        grid.addWidget(QLabel("File (0 deg / isotropic)"), 3, 0, Qt.AlignLeft)
        file0_edit = QLineEdit()
        bind_line_edit(self.file_0deg_var, file0_edit)
        grid.addWidget(file0_edit, 3, 1)
        browse0 = QPushButton("Browse")
        browse0.clicked.connect(self._browse_0deg)
        grid.addWidget(browse0, 3, 2)

        self.lbl_90 = QLabel("File (90 deg)")
        grid.addWidget(self.lbl_90, 4, 0, Qt.AlignLeft)
        self.ent_90 = QLineEdit()
        bind_line_edit(self.file_90deg_var, self.ent_90)
        grid.addWidget(self.ent_90, 4, 1)
        self.btn_90 = QPushButton("Browse")
        self.btn_90.clicked.connect(self._browse_90deg)
        grid.addWidget(self.btn_90, 4, 2)

        self.lbl_pol = QLabel("Selected principal axis (0 or 90 deg)")
        grid.addWidget(self.lbl_pol, 5, 0, Qt.AlignLeft)
        self.ent_pol = QLineEdit()
        bind_line_edit(self.pol_var, self.ent_pol)
        grid.addWidget(self.ent_pol, 5, 1)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        grid.addWidget(sep, 6, 0, 1, 3)
        grid.addWidget(
            QLabel("Inverse-design thickness range (required to include this layer in a search)"),
            7,
            0,
            1,
            3,
            Qt.AlignLeft,
        )
        grid.addWidget(QLabel("t_min (in)"), 8, 0, Qt.AlignLeft)
        tmin_edit = QLineEdit()
        bind_line_edit(self.inv_t_min_var, tmin_edit)
        grid.addWidget(tmin_edit, 8, 1)
        grid.addWidget(QLabel("t_max (in)"), 9, 0, Qt.AlignLeft)
        tmax_edit = QLineEdit()
        bind_line_edit(self.inv_t_max_var, tmax_edit)
        grid.addWidget(tmax_edit, 9, 1)
        grid.addWidget(QLabel("t_accuracy (in)"), 10, 0, Qt.AlignLeft)
        tacc_edit = QLineEdit()
        bind_line_edit(self.inv_t_acc_var, tacc_edit)
        tacc_edit.setToolTip(
            "Snap searched thickness to this increment (e.g. 0.001 in). "
            "Blank = continuous."
        )
        grid.addWidget(tacc_edit, 10, 1)

        grid.setColumnStretch(1, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)

        outer = QVBoxLayout(self)
        outer.addLayout(grid)
        outer.addWidget(buttons)

        self._sync_state()

    def _sync_state(self) -> None:
        enabled = self.aniso_var.get()
        for widget in (self.lbl_90, self.ent_90, self.btn_90, self.lbl_pol, self.ent_pol):
            widget.setEnabled(enabled)

    def _browse_0deg(self) -> None:
        p = filedialog.askopenfilename(title="Select 0 deg/isotropic property file", parent=self, filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if p:
            self.file_0deg_var.set(p)

    def _browse_90deg(self) -> None:
        p = filedialog.askopenfilename(title="Select 90 deg property file", parent=self, filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if p:
            self.file_90deg_var.set(p)

    def _apply_preset(self) -> None:
        name = self.preset_var.get().strip()
        if not name:
            return
        path = self.presets.get(name)
        if not path:
            return
        self.file_0deg_var.set(path)
        if self.aniso_var.get() and not self.file_90deg_var.get().strip():
            self.file_90deg_var.set(path)

    def _on_ok(self) -> None:
        try:
            thickness_in = float(self.thickness_var.get().strip())
            if thickness_in <= 0:
                raise ValueError("Thickness must be > 0.")
            anisotropic = self.aniso_var.get()
            file_0deg = self.file_0deg_var.get().strip()
            file_90deg = self.file_90deg_var.get().strip()
            polarization_deg = float(self.pol_var.get().strip()) if anisotropic else 0.0
            if anisotropic:
                axis = polarization_deg % 180.0
                if not (
                    min(abs(axis), abs(axis - 180.0)) <= 1e-9
                    or abs(axis - 90.0) <= 1e-9
                ):
                    raise ValueError(
                        "Directional layers require a measured principal axis "
                        "of exactly 0 or 90 deg. Arbitrary tensor rotation is "
                        "not supported by the scalar transmission-line model."
                    )

            if not file_0deg:
                raise ValueError("0 deg/isotropic file is required.")
            if anisotropic and not file_90deg:
                raise ValueError("90 deg file is required for anisotropic layer.")

            inv_t_min_in = _parse_optional_thickness(self.inv_t_min_var.get(), "inv_t_min")
            inv_t_max_in = _parse_optional_thickness(self.inv_t_max_var.get(), "inv_t_max")
            inv_t_accuracy_in = _parse_optional_thickness(
                self.inv_t_acc_var.get(), "inv_t_accuracy"
            )
            if (inv_t_min_in is None) != (inv_t_max_in is None):
                raise ValueError("Set both inv_t_min and inv_t_max, or leave both blank.")
            if (
                inv_t_min_in is not None
                and inv_t_max_in is not None
                and inv_t_max_in < inv_t_min_in
            ):
                raise ValueError("inv_t_max must be >= inv_t_min.")

            self.result = LayerConfig(
                thickness_in=thickness_in,
                anisotropic=anisotropic,
                file_0deg=file_0deg,
                file_90deg=file_90deg,
                polarization_deg=polarization_deg,
                inv_t_min_in=inv_t_min_in,
                inv_t_max_in=inv_t_max_in,
                inv_t_accuracy_in=inv_t_accuracy_in,
            )
            self.accept()
        except Exception as exc:
            messagebox.showerror("Invalid Layer", str(exc), parent=self)


class SheetDialog(QDialog):
    """Dialog for adding or editing a resistive sheet."""

    def __init__(
        self,
        parent: QWidget | None = None,
        initial: LayerConfig | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Resistive Sheet")
        self.setModal(True)
        self.result: LayerConfig | None = None

        init = initial or LayerConfig(
            thickness_in=0.0,
            anisotropic=False,
            file_0deg="",
            file_90deg="",
            polarization_deg=0.0,
            is_sheet=True,
            sheet_resistance=377.0,
        )

        self.rs_var = StringVar(f"{init.sheet_resistance:g}")
        self.inv_rs_min_var = StringVar(
            "" if init.inv_rs_min is None else f"{init.inv_rs_min:g}"
        )
        self.inv_rs_max_var = StringVar(
            "" if init.inv_rs_max is None else f"{init.inv_rs_max:g}"
        )
        self.inv_rs_acc_var = StringVar(
            "" if init.inv_rs_accuracy is None else f"{init.inv_rs_accuracy:g}"
        )

        grid = QGridLayout()
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        grid.addWidget(QLabel("Sheet resistance (\u03a9/sq)"), 0, 0, Qt.AlignLeft)
        rs_edit = QLineEdit()
        bind_line_edit(self.rs_var, rs_edit)
        rs_edit.setToolTip("Nominal resistance; used when no search range is set.")
        grid.addWidget(rs_edit, 0, 1)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        grid.addWidget(sep, 1, 0, 1, 2)
        grid.addWidget(
            QLabel("Inverse-design resistance range (optional; blank = keep fixed)"),
            2,
            0,
            1,
            2,
            Qt.AlignLeft,
        )
        grid.addWidget(QLabel("R min (\u03a9/sq)"), 3, 0, Qt.AlignLeft)
        rmin_edit = QLineEdit()
        bind_line_edit(self.inv_rs_min_var, rmin_edit)
        grid.addWidget(rmin_edit, 3, 1)
        grid.addWidget(QLabel("R max (\u03a9/sq)"), 4, 0, Qt.AlignLeft)
        rmax_edit = QLineEdit()
        bind_line_edit(self.inv_rs_max_var, rmax_edit)
        grid.addWidget(rmax_edit, 4, 1)
        grid.addWidget(QLabel("R accuracy (\u03a9)"), 5, 0, Qt.AlignLeft)
        racc_edit = QLineEdit()
        bind_line_edit(self.inv_rs_acc_var, racc_edit)
        racc_edit.setToolTip(
            "Snap searched resistance to this increment (e.g. 1 ohm). Blank = continuous."
        )
        grid.addWidget(racc_edit, 5, 1)
        grid.setColumnStretch(1, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)

        outer = QVBoxLayout(self)
        outer.addLayout(grid)
        outer.addWidget(buttons)

    def _on_ok(self) -> None:
        try:
            rs = float(self.rs_var.get().strip())
            if rs <= 0:
                raise ValueError("Sheet resistance must be > 0.")
            inv_rs_min = _parse_optional_thickness(self.inv_rs_min_var.get(), "R min")
            inv_rs_max = _parse_optional_thickness(self.inv_rs_max_var.get(), "R max")
            inv_rs_accuracy = _parse_optional_thickness(
                self.inv_rs_acc_var.get(), "R accuracy"
            )
            if (inv_rs_min is None) != (inv_rs_max is None):
                raise ValueError("Set both R min and R max, or leave both blank.")
            if (
                inv_rs_min is not None
                and inv_rs_max is not None
                and inv_rs_max < inv_rs_min
            ):
                raise ValueError("R max must be >= R min.")
            self.result = LayerConfig(
                thickness_in=0.0,
                anisotropic=False,
                file_0deg="",
                file_90deg="",
                polarization_deg=0.0,
                is_sheet=True,
                sheet_resistance=rs,
                inv_rs_min=inv_rs_min,
                inv_rs_max=inv_rs_max,
                inv_rs_accuracy=inv_rs_accuracy,
            )
            self.accept()
        except Exception as exc:
            messagebox.showerror("Invalid Sheet", str(exc), parent=self)


class MixComponentDialog(QDialog):
    """Dialog for adding or editing one component of a material blend."""

    def __init__(
        self,
        parent: QWidget | None = None,
        initial: dict | None = None,
        presets: dict[str, str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Mix Component")
        self.setModal(True)
        self.result: dict | None = None
        self.presets = presets or {}

        init = initial or {
            "file": "",
            "parts": 50.0,
            "min": 0.0,
            "max": 100.0,
            "density": 0.0,
            "units": "volume_percent",
        }
        self.file_var = StringVar(str(init.get("file", "")))
        self.parts_var = StringVar(f"{float(init.get('parts', 1.0)):g}")
        self.min_var = StringVar(f"{float(init.get('min', 0.0)):g}")
        self.max_var = StringVar(f"{float(init.get('max', 3.0)):g}")
        init_density = float(init.get("density", 0.0))
        self.density_var = StringVar(f"{init_density:g}" if init_density > 0 else "")
        self.preset_var = StringVar("")

        grid = QGridLayout()
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        grid.addWidget(QLabel("Preset material"), 0, 0, Qt.AlignLeft)
        preset_values = [""] + sorted(self.presets.keys())
        self.preset_combo = make_combo(preset_values, self.preset_var, width=220)
        grid.addWidget(self.preset_combo, 0, 1)
        use_btn = QPushButton("Use")
        use_btn.clicked.connect(self._apply_preset)
        grid.addWidget(use_btn, 0, 2)

        grid.addWidget(QLabel("Property file"), 1, 0, Qt.AlignLeft)
        file_edit = QLineEdit()
        bind_line_edit(self.file_var, file_edit)
        grid.addWidget(file_edit, 1, 1)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse)
        grid.addWidget(browse, 1, 2)

        recipe_help = QLabel(
            "Mixing laws use volume fraction. Enter relative volume amounts "
            "for the known recipe (for example 30 and 70). FREDDY normalizes "
            "all component amounts to 100%."
        )
        recipe_help.setWordWrap(True)
        grid.addWidget(recipe_help, 2, 0, 1, 3)

        grid.addWidget(QLabel("Recipe volume amount"), 3, 0, Qt.AlignLeft)
        parts_edit = QLineEdit()
        bind_line_edit(self.parts_var, parts_edit)
        grid.addWidget(parts_edit, 3, 1)

        grid.addWidget(QLabel("Density (g/cc, optional)"), 4, 0, Qt.AlignLeft)
        density_edit = QLineEdit()
        density_edit.setPlaceholderText("blank = unknown")
        bind_line_edit(self.density_var, density_edit)
        grid.addWidget(density_edit, 4, 1)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        grid.addWidget(sep, 5, 0, 1, 3)
        grid.addWidget(
            QLabel("Allowed volume fraction when finding a recipe"),
            6,
            0,
            1,
            3,
            Qt.AlignLeft,
        )
        grid.addWidget(QLabel("Minimum (vol %)"), 7, 0, Qt.AlignLeft)
        min_edit = QLineEdit()
        bind_line_edit(self.min_var, min_edit)
        grid.addWidget(min_edit, 7, 1)
        grid.addWidget(QLabel("Maximum (vol %)"), 8, 0, Qt.AlignLeft)
        max_edit = QLineEdit()
        bind_line_edit(self.max_var, max_edit)
        grid.addWidget(max_edit, 8, 1)
        grid.setColumnStretch(1, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        outer = QVBoxLayout(self)
        outer.addLayout(grid)
        outer.addWidget(buttons)

    def _browse(self) -> None:
        p = filedialog.askopenfilename(title="Select property file", parent=self, filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if p:
            self.file_var.set(p)

    def _apply_preset(self) -> None:
        name = self.preset_var.get().strip()
        if not name:
            return
        path = self.presets.get(name)
        if path:
            self.file_var.set(path)

    def _on_ok(self) -> None:
        try:
            file_str = self.file_var.get().strip()
            if not file_str:
                raise ValueError("A property file is required.")
            parts = float(self.parts_var.get().strip())
            pmin = float(self.min_var.get().strip())
            pmax = float(self.max_var.get().strip())
            if not all(math.isfinite(value) for value in (parts, pmin, pmax)):
                raise ValueError("Recipe amount and search bounds must be finite.")
            if parts < 0 or pmin < 0 or pmax < 0:
                raise ValueError("Recipe amount and volume bounds must be >= 0.")
            if pmin > 100 or pmax > 100:
                raise ValueError("Volume-fraction bounds cannot exceed 100%.")
            if pmax < pmin:
                raise ValueError("Maximum volume % must be >= minimum volume %.")
            density_str = self.density_var.get().strip()
            density = float(density_str) if density_str else 0.0
            if not math.isfinite(density) or density < 0:
                raise ValueError("Density must be >= 0 (blank or 0 = unknown).")
            # Fail immediately on a bad schema/passivity convention rather
            # than waiting until a long recipe search starts.
            read_material_table(Path(file_str), 0)
            self.result = {
                "file": file_str,
                "parts": parts,
                "min": pmin,
                "max": pmax,
                "density": density,
                "units": "volume_percent",
            }
            self.accept()
        except Exception as exc:
            messagebox.showerror("Invalid Component", str(exc), parent=self)


class ImpedanceGui(QMainWindow):
    # Host integrations may consume this deliberately narrow artifact stream.
    # It is never emitted for off-angle/thickness analysis, uncertainty, or a
    # multi-file IBC batch with no unambiguous current file. ``kind`` is exactly
    # ``ibc`` or ``material`` and the second value is the absolute path of one
    # solver-compatible nominal CSV.
    nominal_artifact_exported = Signal(str, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the FREDDY workspace as a window or an embedded child widget."""
        super().__init__(parent)
        self.setWindowTitle(APP_TITLE)
        self.resize(1180, 820)
        self.setMinimumSize(960, 640)

        self.layers: list[LayerConfig] = []

        self.f_start_var = StringVar("1.0")
        self.f_stop_var = StringVar("18.0")
        self.f_step_var = StringVar("0.1")
        # A PEC-backed front-face input impedance is the safe default for
        # collapsing a coating onto a Type 2 RCS body.
        self.backing_var = StringVar("pec")
        self.output_var = StringVar("impedance_out.csv")
        # Nominal IBC batch: one PEC-backed broadside solver CSV per selected-
        # layer thickness. The frequency grid is deliberately shared with the
        # single Impedance mode above.
        self.ibc_batch_layer_var = StringVar("")
        self.ibc_batch_start_var = StringVar("15")
        self.ibc_batch_stop_var = StringVar("30")
        self.ibc_batch_step_var = StringVar("1")
        self.ibc_batch_unit_var = StringVar("mil")
        self.ibc_batch_output_dir_var = StringVar(".")
        self.ibc_batch_prefix_var = StringVar("ibc")
        self.uncertainty_var = BooleanVar(False)
        self.unc_t_pct_var = StringVar("5.0")
        self.unc_eps_pct_var = StringVar("5.0")
        self.unc_mu_pct_var = StringVar("5.0")
        # Off Angle tab keeps its own frequency sweep, output, and uncertainty.
        self.angle_f_start_var = StringVar("1.0")
        self.angle_f_stop_var = StringVar("18.0")
        self.angle_f_step_var = StringVar("0.1")
        self.angle_start_var = StringVar("0.0")
        self.angle_stop_var = StringVar("80.0")
        self.angle_step_var = StringVar("1.0")
        self.wave_pol_var = StringVar("TE")
        self.angle_output_var = StringVar("angle_out.csv")
        self.angle_uncertainty_var = BooleanVar(False)
        self.angle_unc_t_pct_var = StringVar("5.0")
        self.angle_unc_eps_pct_var = StringVar("5.0")
        self.angle_unc_mu_pct_var = StringVar("5.0")
        # Thickness tab: sweep one layer's thickness against frequency at a
        # fixed incidence angle. Every other layer stays as configured.
        self.thk_f_start_var = StringVar("1.0")
        self.thk_f_stop_var = StringVar("18.0")
        self.thk_f_step_var = StringVar("0.1")
        self.thk_start_var = StringVar("0.01")
        self.thk_stop_var = StringVar("0.25")
        self.thk_step_var = StringVar("0.005")
        self.thk_layer_var = StringVar("")
        self.thk_angle_var = StringVar("0.0")
        self.thk_wave_pol_var = StringVar("TE")
        self.thk_output_var = StringVar("thickness_out.csv")
        self.thk_uncertainty_var = BooleanVar(False)
        self.thk_unc_t_pct_var = StringVar("5.0")
        self.thk_unc_eps_pct_var = StringVar("5.0")
        self.thk_unc_mu_pct_var = StringVar("5.0")
        self.heatmap_metric_var = StringVar(HEATMAP_METRIC_OPTIONS[0][0])
        self.uncertainty_view_var = StringVar(UNCERTAINTY_VIEW_OPTIONS[0][0])
        self.metric_label_to_key = {label: key for label, key in HEATMAP_METRIC_OPTIONS}
        self.metric_key_to_label = {key: label for label, key in HEATMAP_METRIC_OPTIONS}
        self.uncertainty_view_label_to_key = {label: key for label, key in UNCERTAINTY_VIEW_OPTIONS}
        self.cbar_auto_var = BooleanVar(True)
        self.cbar_min_var = StringVar("")
        self.cbar_max_var = StringVar("")
        self.slice_angle_var = StringVar("")
        self.slice_freq_var = StringVar("")
        self.inv_freq_mode_var = StringVar("Band sweep")
        self.inv_freq_list_var = StringVar("8.0, 10.0, 12.0")
        self.inv_target_start_var = StringVar("8.0")
        self.inv_target_stop_var = StringVar("12.0")
        self.inv_target_step_var = StringVar("0.25")
        self.inv_angle_start_var = StringVar("0.0")
        self.inv_angle_stop_var = StringVar("80.0")
        self.inv_angle_step_var = StringVar("5.0")
        self.inv_wave_pol_var = StringVar("TE")
        self.inv_max_evals_var = StringVar("400")
        self.inv_top_n_var = StringVar("10")
        self.inv_percentile_var = StringVar("10")
        self.inv_uncertainty_var = BooleanVar(True)
        self.inv_unc_t_pct_var = StringVar("5.0")
        self.inv_unc_eps_pct_var = StringVar("5.0")
        self.inv_unc_mu_pct_var = StringVar("5.0")
        self.inv_score_mode_var = StringVar(INVERSE_SCORE_MODE_OPTIONS[0])
        self.inv_refine_var = BooleanVar(True)
        self.inv_seed_var = StringVar("")
        # Material Mix tab: predict effective properties from a volume recipe or
        # find bounded volume-fraction recipes for properties/planar performance.
        self.mix_components: list[dict] = []
        self.mix_rule_var = StringVar(MIX_RULE_LABEL_OPTIONS[0])
        self.mix_objective_var = StringVar(MIX_OBJECTIVE_OPTIONS[0])
        self.mix_thickness_var = StringVar("0.125")
        self.mix_freq_mode_var = StringVar("Band sweep")
        self.mix_freq_list_var = StringVar("8.0, 10.0, 12.0")
        self.mix_target_start_var = StringVar("0.1")
        self.mix_target_stop_var = StringVar("18.0")
        self.mix_target_step_var = StringVar("0.1")
        # Property-design target: constants or a measured 5-column material
        # file, plus eps/mu weighting for the match error.
        self.mix_prop_source_var = StringVar(MIX_PROP_SOURCE_OPTIONS[0])
        self.mix_prop_eps_re_var = StringVar("7.0")
        self.mix_prop_eps_im_var = StringVar("-0.5")
        self.mix_prop_mu_re_var = StringVar("1.0")
        self.mix_prop_mu_im_var = StringVar("0.0")
        self.mix_prop_file_var = StringVar("")
        self.mix_prop_weps_var = StringVar("1.0")
        self.mix_prop_wmu_var = StringVar("1.0")
        self.mix_perf_metric_var = StringVar(MIX_PERFORMANCE_METRIC_OPTIONS[0][0])
        self.mix_perf_target_var = StringVar("-10.0")
        self.mix_perf_angle_start_var = StringVar("0.0")
        self.mix_perf_angle_stop_var = StringVar("60.0")
        self.mix_perf_angle_step_var = StringVar("5.0")
        self.mix_perf_wave_pol_var = StringVar("TE")
        self.mix_max_evals_var = StringVar("400")
        self.mix_top_n_var = StringVar("10")
        self.mix_seed_var = StringVar("")
        self.mix_refine_var = BooleanVar(True)
        self.mix_score_mode_var = StringVar(MIX_SCORE_MODE_OPTIONS[0])
        self.mix_uncertainty_var = BooleanVar(False)
        self.mix_unc_t_pct_var = StringVar("5.0")
        self.mix_unc_eps_pct_var = StringVar("5.0")
        self.mix_unc_mu_pct_var = StringVar("5.0")
        self.dark_mode_var = BooleanVar(True)
        self.project_path: Path | None = None
        self.inverse_candidates: list[InverseCandidate] = []
        self._colors = DARK_THEME

        # Plot objects are created in _build_ui(). Initialize here so early callbacks are safe.
        self.fig = None
        self.ax_heatmap = None
        self.ax_freq_slice = None
        self.ax_angle_slice = None
        self.canvas = None
        self.plot_frame = None
        self.heatmap_cbar = None
        self.heatmap_click_cid = None
        # Slice selection is shared by the Off Angle and Thickness heatmaps;
        # x is angle or thickness depending on which tab is active.
        self.selected_x_idx: int | None = None
        self.selected_freq_idx: int | None = None
        self.inv_results_list: QListWidget | None = None
        self.left_tabs = None
        self.mode_stack = None
        self.nav_group = None
        self.dark_mode_action = None
        self._mode_labels: list[str] = []
        self.angle_tab = None
        self.thickness_tab = None
        self.thk_layer_combo = None
        self.ibc_batch_layer_combo = None
        self.ibc_batch_preview_label = None
        self.ibc_batch_export_btn = None
        self.thk_unc_details_frame = None
        self.thk_unc_t_entry = None
        self.thk_unc_eps_entry = None
        self.thk_unc_mu_entry = None
        self.slice_x_label = None
        self.inv_tab = None
        self.inv_unc_t_entry = None
        self.inv_unc_eps_entry = None
        self.inv_unc_mu_entry = None
        self.inv_percentile_entry = None
        self.inv_target_start_entry = None
        self.inv_target_stop_entry = None
        self.inv_target_step_entry = None
        self.inv_freq_list_entry = None
        self.layer_add_btn = None
        self.layer_add_sheet_btn = None
        self.layer_edit_btn = None
        self.layer_remove_btn = None
        self.layer_up_btn = None
        self.layer_down_btn = None
        self.compute_btn = None
        self.angle_compute_btn = None
        self.thk_compute_btn = None
        self.inv_run_btn = None
        self.inv_apply_btn = None
        self.status_var = StringVar("Ready")
        self.status_progress = None
        self._task_running = False

        self.last_heatmap_results: dict[str, list[list[float]] | list[float]] | None = None
        self.last_heatmap_uncertainty_min: dict[str, list[list[float]]] | None = None
        self.last_heatmap_uncertainty_max: dict[str, list[list[float]]] | None = None
        self.last_thickness_results: dict[str, list[list[float]] | list[float]] | None = None
        self.last_thickness_uncertainty_min: dict[str, list[list[float]]] | None = None
        self.last_thickness_uncertainty_max: dict[str, list[list[float]]] | None = None
        self.inverse_plot_freqs: list[float] = []
        self.inverse_plot_samples: list[list[list[float]]] = []

        # Material Mix tab widget handles and result state.
        self.mix_tab = None
        self.mix_list: QListWidget | None = None
        self.mix_results_list: QListWidget | None = None
        self.mix_results_frame = None
        self.mix_search_frame = None
        self.mix_model_help_label: QLabel | None = None
        self.mix_workflow_help_label: QLabel | None = None
        self.mix_summary_label: QLabel | None = None
        self.mix_candidates: list[MixCandidate] = []
        self.mix_preview: dict | None = None
        self.mix_plot_data: list[dict] = []
        self.mix_add_btn = None
        self.mix_edit_btn = None
        self.mix_remove_btn = None
        self.mix_run_btn = None
        self.mix_preview_btn = None
        self.mix_apply_btn = None
        self.mix_export_btn = None
        self.mix_target_start_entry = None
        self.mix_target_stop_entry = None
        self.mix_target_step_entry = None
        self.mix_freq_list_entry = None
        self.mix_unc_t_entry = None
        self.mix_unc_eps_entry = None
        self.mix_unc_mu_entry = None
        self.mix_prop_frame = None
        self.mix_perf_frame = None
        self.mix_perf_requirement_label: QLabel | None = None
        self.mix_prop_const_entries: list = []
        self.mix_prop_file_entry = None
        self.mix_prop_browse_btn = None

        self._build_ui()
        self._apply_theme()
        if Path("material.csv").exists():
            self.layers.append(
                LayerConfig(
                    thickness_in=0.125,
                    anisotropic=False,
                    file_0deg="material.csv",
                    file_90deg="",
                    polarization_deg=0.0,
                )
            )
            self._refresh_layers()

    def job_is_running(self) -> bool:
        """Return whether FREDDY currently owns an active background job."""
        return bool(self._task_running)

    def _publish_nominal_artifact(self, kind: str, path: Path | str) -> None:
        """Publish one validated, solver-facing CSV to an embedding host.

        Keeping this whitelist at FREDDY's authoritative export boundary makes
        it impossible for analysis-only CSVs to enter a GHOST material table
        merely because their filenames also end in ``.csv``.
        """

        artifact_kind = str(kind).strip().lower()
        if artifact_kind not in {"ibc", "material"}:
            raise ValueError(
                "Attachable FREDDY artifacts must be nominal IBC or material CSVs."
            )
        artifact_path = Path(path).expanduser().resolve()
        if artifact_path.suffix.lower() != ".csv" or not artifact_path.is_file():
            raise ValueError(
                f"Attachable FREDDY artifact is not a readable CSV: {artifact_path}"
            )
        self.nominal_artifact_exported.emit(
            artifact_kind, str(artifact_path)
        )

    def can_close(self) -> bool:
        """Return whether a host may safely remove or close this workspace."""
        return not self.job_is_running()

    def _confirm_output_replacements(
        self, paths: list[Path], *, operation: str
    ) -> bool:
        """Preflight every output on the GUI thread before starting a worker."""

        unique: list[Path] = []
        seen: set[str] = set()
        for raw_path in paths:
            path = Path(raw_path).expanduser()
            key = os.path.normcase(os.path.abspath(path)).casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)

        invalid = [path for path in unique if path.exists() and not path.is_file()]
        if invalid:
            messagebox.showerror(
                f"{operation} Output",
                "An output path is not a file:\n" + "\n".join(str(p) for p in invalid),
                parent=self,
            )
            return False
        existing = [path for path in unique if path.is_file()]
        if not existing:
            return True
        shown = "\n".join(str(path.resolve()) for path in existing[:10])
        if len(existing) > 10:
            shown += f"\n…and {len(existing) - 10} more"
        return messagebox.askyesno(
            f"Replace Existing {operation} Output?",
            f"{len(existing)} output file(s) already exist:\n\n{shown}\n\n"
            "Replace all listed files?",
            parent=self,
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt callback name
        """Keep standalone FREDDY alive until its background write finishes."""
        if self.job_is_running():
            QMessageBox.warning(
                self,
                "FREDDY Task Still Running",
                "A FREDDY material or IBC task is still running. Wait for it "
                "to finish before closing FREDDY.",
            )
            event.ignore()
            return
        super().closeEvent(event)

    def _build_ui(self) -> None:
        def _entry(var: StringVar, chars: int | None = None) -> QLineEdit:
            edit = QLineEdit()
            bind_line_edit(var, edit)
            if chars is not None:
                edit.setMaximumWidth(chars * 9 + 16)
            return edit

        # Global actions live in a menu bar (File / View / Help) rather than an
        # inline button row, and the window is organized as a left navigation
        # rail driving a stacked workspace above a full-width results band.
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        load_action = QAction("Load Project…", self)
        load_action.triggered.connect(self._load_project)
        file_menu.addAction(load_action)
        save_action = QAction("Save Project…", self)
        save_action.triggered.connect(self._save_project)
        file_menu.addAction(save_action)
        view_menu = menubar.addMenu("View")
        self.dark_mode_action = QAction("Dark mode", self)
        self.dark_mode_action.setCheckable(True)
        self.dark_mode_action.setChecked(self.dark_mode_var.get())
        self.dark_mode_action.toggled.connect(self.dark_mode_var.set)
        self.dark_mode_var.valueChanged.connect(self.dark_mode_action.setChecked)
        self.dark_mode_var.valueChanged.connect(lambda _v: self._apply_theme())
        view_menu.addAction(self.dark_mode_action)
        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        nav = QFrame()
        nav.setObjectName("NavRail")
        nav.setFixedWidth(166)
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(10, 14, 10, 14)
        nav_layout.setSpacing(4)
        brand = QLabel(APP_ACRONYM)
        brand.setObjectName("NavBrand")
        nav_layout.addWidget(brand)
        nav_layout.addSpacing(10)
        root_layout.addWidget(nav)

        self.mode_stack = QStackedWidget()
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self._mode_labels = []

        def _add_mode(label: str, page: QWidget) -> None:
            index = self.mode_stack.count()
            self.mode_stack.addWidget(page)
            button = QToolButton()
            button.setObjectName("ModeNavButton")
            button.setText(label)
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonTextOnly)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            if index == 0:
                button.setChecked(True)
            self.nav_group.addButton(button, index)
            nav_layout.addWidget(button)
            self._mode_labels.append(label)

        def _output_row(
            var: StringVar,
            browse_cb: Callable[[], None],
            label: str = "Output file",
        ) -> QWidget:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(QLabel(label))
            row_layout.addWidget(_entry(var), 1)
            browse = QPushButton("Browse")
            browse.clicked.connect(browse_cb)
            row_layout.addWidget(browse)
            return row

        def _uncertainty_group(enabled_var, t_var, eps_var, mu_var, sync_cb):
            group = QGroupBox("Uncertainty corners")
            grid = QGridLayout(group)
            check = QCheckBox(
                "Enable (nominal solver CSV stays 3-column; bounds use a separate report)"
            )
            bind_check_box(enabled_var, check)
            check.clicked.connect(sync_cb)
            grid.addWidget(check, 0, 0, 1, 6, Qt.AlignLeft)
            details = QWidget()
            dgrid = QGridLayout(details)
            dgrid.setContentsMargins(0, 0, 0, 0)
            dgrid.addWidget(QLabel("Thickness ±%"), 0, 0, Qt.AlignLeft)
            t_entry = _entry(t_var, 8)
            dgrid.addWidget(t_entry, 0, 1, Qt.AlignLeft)
            dgrid.addWidget(QLabel("Eps ±%"), 0, 2, Qt.AlignLeft)
            eps_entry = _entry(eps_var, 8)
            dgrid.addWidget(eps_entry, 0, 3, Qt.AlignLeft)
            dgrid.addWidget(QLabel("Mu ±%"), 0, 4, Qt.AlignLeft)
            mu_entry = _entry(mu_var, 8)
            dgrid.addWidget(mu_entry, 0, 5, Qt.AlignLeft)
            dgrid.setColumnStretch(6, 1)
            grid.addWidget(details, 1, 0, 1, 6)
            return group, details, t_entry, eps_entry, mu_entry

        # --- Impedance mode: frequency sweep + backing -> freq/R/X file ---
        imp_tab = QWidget()
        imp_layout = QVBoxLayout(imp_tab)
        _add_mode("Impedance", imp_tab)

        imp_freq_group = QGroupBox("Frequency sweep")
        imp_freq_grid = QGridLayout(imp_freq_group)
        imp_freq_grid.addWidget(QLabel("Start (GHz)"), 0, 0, Qt.AlignLeft)
        imp_freq_grid.addWidget(_entry(self.f_start_var, 10), 0, 1, Qt.AlignLeft)
        imp_freq_grid.addWidget(QLabel("Stop"), 0, 2, Qt.AlignLeft)
        imp_freq_grid.addWidget(_entry(self.f_stop_var, 10), 0, 3, Qt.AlignLeft)
        imp_freq_grid.addWidget(QLabel("Step"), 0, 4, Qt.AlignLeft)
        imp_freq_grid.addWidget(_entry(self.f_step_var, 10), 0, 5, Qt.AlignLeft)
        self.backing_label = QLabel("Backing")
        imp_freq_grid.addWidget(self.backing_label, 1, 0, Qt.AlignLeft)
        self.backing_combo = make_combo(("pec", "air"), self.backing_var, width=110)
        self.backing_combo.setToolTip(
            "PEC: front-face impedance of a coating on a conductor; suitable "
            "for collapsing the coating onto a Type 2 RCS body.\n"
            "air: planar air-terminated input impedance for analysis; it is "
            "not a general one-sided IBC for a closed transmitting body."
        )
        imp_freq_grid.addWidget(self.backing_combo, 1, 1, 1, 3, Qt.AlignLeft)
        imp_freq_grid.setColumnStretch(5, 1)
        imp_layout.addWidget(imp_freq_group)

        (
            imp_unc_group,
            self.unc_details_frame,
            self.unc_t_entry,
            self.unc_eps_entry,
            self.unc_mu_entry,
        ) = _uncertainty_group(
            self.uncertainty_var,
            self.unc_t_pct_var,
            self.unc_eps_pct_var,
            self.unc_mu_pct_var,
            self._sync_uncertainty_state,
        )
        imp_layout.addWidget(imp_unc_group)
        imp_layout.addWidget(
            _output_row(
                self.output_var,
                self._browse_output,
                "Nominal 2D/BoR IBC CSV",
            )
        )

        imp_btn_row = QHBoxLayout()
        self.compute_btn = QPushButton("Compute")
        self.compute_btn.clicked.connect(self._compute_impedance)
        imp_btn_row.addWidget(self.compute_btn)
        imp_btn_row.addStretch(1)
        imp_layout.addLayout(imp_btn_row)
        imp_layout.addStretch(1)

        # --- IBC Batch: one nominal PEC-backed IBC per layer thickness ---
        ibc_batch_tab = QWidget()
        ibc_batch_layout = QVBoxLayout(ibc_batch_tab)
        _add_mode("IBC Batch", ibc_batch_tab)

        ibc_batch_intro = QLabel(
            "Create one solver-compatible three-column IBC CSV per thickness. "
            "Only the selected material layer changes; the stack order, other "
            "layers, and material data stay fixed. Every file is broadside and "
            "PEC-backed."
        )
        ibc_batch_intro.setWordWrap(True)
        ibc_batch_layout.addWidget(ibc_batch_intro)

        ibc_batch_freq_group = QGroupBox("Frequency sweep (shared with Impedance)")
        ibc_batch_freq_grid = QGridLayout(ibc_batch_freq_group)
        ibc_batch_freq_grid.addWidget(QLabel("Start (GHz)"), 0, 0, Qt.AlignLeft)
        ibc_batch_freq_grid.addWidget(_entry(self.f_start_var, 10), 0, 1, Qt.AlignLeft)
        ibc_batch_freq_grid.addWidget(QLabel("Stop"), 0, 2, Qt.AlignLeft)
        ibc_batch_freq_grid.addWidget(_entry(self.f_stop_var, 10), 0, 3, Qt.AlignLeft)
        ibc_batch_freq_grid.addWidget(QLabel("Step"), 0, 4, Qt.AlignLeft)
        ibc_batch_freq_grid.addWidget(_entry(self.f_step_var, 10), 0, 5, Qt.AlignLeft)
        ibc_batch_freq_grid.setColumnStretch(5, 1)
        ibc_batch_layout.addWidget(ibc_batch_freq_group)

        ibc_batch_sweep_group = QGroupBox("Selected-layer thickness sweep")
        ibc_batch_sweep_grid = QGridLayout(ibc_batch_sweep_group)
        ibc_batch_sweep_grid.addWidget(QLabel("Layer"), 0, 0, Qt.AlignLeft)
        self.ibc_batch_layer_combo = make_combo(
            (), self.ibc_batch_layer_var, width=260
        )
        ibc_batch_sweep_grid.addWidget(
            self.ibc_batch_layer_combo, 0, 1, 1, 5, Qt.AlignLeft
        )
        ibc_batch_sweep_grid.addWidget(QLabel("Start"), 1, 0, Qt.AlignLeft)
        ibc_batch_sweep_grid.addWidget(
            _entry(self.ibc_batch_start_var, 10), 1, 1, Qt.AlignLeft
        )
        ibc_batch_sweep_grid.addWidget(QLabel("Stop"), 1, 2, Qt.AlignLeft)
        ibc_batch_sweep_grid.addWidget(
            _entry(self.ibc_batch_stop_var, 10), 1, 3, Qt.AlignLeft
        )
        ibc_batch_sweep_grid.addWidget(QLabel("Step"), 1, 4, Qt.AlignLeft)
        ibc_batch_sweep_grid.addWidget(
            _entry(self.ibc_batch_step_var, 10), 1, 5, Qt.AlignLeft
        )
        ibc_batch_sweep_grid.addWidget(QLabel("Units"), 2, 0, Qt.AlignLeft)
        ibc_batch_sweep_grid.addWidget(
            make_combo(("mil", "in", "mm"), self.ibc_batch_unit_var, width=80),
            2,
            1,
            Qt.AlignLeft,
        )
        ibc_batch_sweep_grid.setColumnStretch(5, 1)
        ibc_batch_layout.addWidget(ibc_batch_sweep_group)

        ibc_batch_output_group = QGroupBox("Output naming")
        ibc_batch_output_grid = QGridLayout(ibc_batch_output_group)
        ibc_batch_output_grid.addWidget(QLabel("Folder"), 0, 0, Qt.AlignLeft)
        ibc_batch_output_grid.addWidget(
            _entry(self.ibc_batch_output_dir_var), 0, 1, 1, 4
        )
        ibc_batch_browse = QPushButton("Browse")
        ibc_batch_browse.clicked.connect(self._browse_ibc_batch_output_dir)
        ibc_batch_output_grid.addWidget(ibc_batch_browse, 0, 5)
        ibc_batch_output_grid.addWidget(QLabel("File prefix"), 1, 0, Qt.AlignLeft)
        ibc_batch_output_grid.addWidget(
            _entry(self.ibc_batch_prefix_var), 1, 1, 1, 2
        )
        ibc_batch_output_grid.addWidget(
            QLabel("Pattern: <prefix>_<thickness><unit>.csv"),
            1,
            3,
            1,
            3,
            Qt.AlignLeft,
        )
        ibc_batch_layout.addWidget(ibc_batch_output_group)

        self.ibc_batch_preview_label = QLabel("")
        self.ibc_batch_preview_label.setWordWrap(True)
        ibc_batch_layout.addWidget(self.ibc_batch_preview_label)
        self.ibc_batch_export_btn = QPushButton("Export IBC batch")
        self.ibc_batch_export_btn.clicked.connect(self._export_ibc_batch)
        ibc_batch_layout.addWidget(self.ibc_batch_export_btn, 0, Qt.AlignLeft)
        ibc_batch_layout.addStretch(1)

        for batch_var in (
            self.ibc_batch_layer_var,
            self.ibc_batch_start_var,
            self.ibc_batch_stop_var,
            self.ibc_batch_step_var,
            self.ibc_batch_unit_var,
            self.ibc_batch_output_dir_var,
            self.ibc_batch_prefix_var,
        ):
            batch_var.valueChanged.connect(
                lambda _value: self._refresh_ibc_batch_preview()
            )

        # --- Off Angle tab: frequency x angle heatmap ---
        angle_tab = QWidget()
        self.angle_tab = angle_tab
        angle_layout = QVBoxLayout(angle_tab)
        _add_mode("Off Angle", angle_tab)

        ang_freq_group = QGroupBox("Frequency sweep")
        ang_freq_grid = QGridLayout(ang_freq_group)
        ang_freq_grid.addWidget(QLabel("Start (GHz)"), 0, 0, Qt.AlignLeft)
        ang_freq_grid.addWidget(_entry(self.angle_f_start_var, 10), 0, 1, Qt.AlignLeft)
        ang_freq_grid.addWidget(QLabel("Stop"), 0, 2, Qt.AlignLeft)
        ang_freq_grid.addWidget(_entry(self.angle_f_stop_var, 10), 0, 3, Qt.AlignLeft)
        ang_freq_grid.addWidget(QLabel("Step"), 0, 4, Qt.AlignLeft)
        ang_freq_grid.addWidget(_entry(self.angle_f_step_var, 10), 0, 5, Qt.AlignLeft)
        ang_freq_grid.setColumnStretch(5, 1)
        angle_layout.addWidget(ang_freq_group)

        angle_group = QGroupBox("Angle sweep")
        angle_grid = QGridLayout(angle_group)
        angle_grid.addWidget(QLabel("Start (deg)"), 0, 0, Qt.AlignLeft)
        angle_grid.addWidget(_entry(self.angle_start_var, 10), 0, 1, Qt.AlignLeft)
        angle_grid.addWidget(QLabel("Stop"), 0, 2, Qt.AlignLeft)
        angle_grid.addWidget(_entry(self.angle_stop_var, 10), 0, 3, Qt.AlignLeft)
        angle_grid.addWidget(QLabel("Step"), 0, 4, Qt.AlignLeft)
        angle_grid.addWidget(_entry(self.angle_step_var, 10), 0, 5, Qt.AlignLeft)
        angle_grid.addWidget(QLabel("Wave pol"), 1, 0, Qt.AlignLeft)
        angle_grid.addWidget(make_combo(("TE", "TM"), self.wave_pol_var, width=70), 1, 1, Qt.AlignLeft)
        angle_grid.setColumnStretch(5, 1)
        angle_layout.addWidget(angle_group)

        (
            ang_unc_group,
            self.angle_unc_details_frame,
            self.angle_unc_t_entry,
            self.angle_unc_eps_entry,
            self.angle_unc_mu_entry,
        ) = _uncertainty_group(
            self.angle_uncertainty_var,
            self.angle_unc_t_pct_var,
            self.angle_unc_eps_pct_var,
            self.angle_unc_mu_pct_var,
            self._sync_angle_uncertainty_state,
        )
        angle_layout.addWidget(ang_unc_group)
        angle_layout.addWidget(_output_row(self.angle_output_var, self._browse_angle_output))

        self.angle_compute_btn = QPushButton("Compute")
        self.angle_compute_btn.clicked.connect(self._compute_off_angle)
        angle_layout.addWidget(self.angle_compute_btn, 0, Qt.AlignLeft)
        angle_layout.addStretch(1)

        # --- Thickness tab: thickness x frequency heatmap for one layer ---
        thickness_tab = QWidget()
        self.thickness_tab = thickness_tab
        thk_layout = QVBoxLayout(thickness_tab)
        _add_mode("Thickness", thickness_tab)

        thk_freq_group = QGroupBox("Frequency sweep")
        thk_freq_grid = QGridLayout(thk_freq_group)
        thk_freq_grid.addWidget(QLabel("Start (GHz)"), 0, 0, Qt.AlignLeft)
        thk_freq_grid.addWidget(_entry(self.thk_f_start_var, 10), 0, 1, Qt.AlignLeft)
        thk_freq_grid.addWidget(QLabel("Stop"), 0, 2, Qt.AlignLeft)
        thk_freq_grid.addWidget(_entry(self.thk_f_stop_var, 10), 0, 3, Qt.AlignLeft)
        thk_freq_grid.addWidget(QLabel("Step"), 0, 4, Qt.AlignLeft)
        thk_freq_grid.addWidget(_entry(self.thk_f_step_var, 10), 0, 5, Qt.AlignLeft)
        thk_freq_grid.setColumnStretch(5, 1)
        thk_layout.addWidget(thk_freq_group)

        thk_group = QGroupBox("Thickness sweep")
        thk_grid = QGridLayout(thk_group)
        thk_grid.addWidget(QLabel("Layer"), 0, 0, Qt.AlignLeft)
        self.thk_layer_combo = make_combo((), self.thk_layer_var, width=260)
        thk_grid.addWidget(self.thk_layer_combo, 0, 1, 1, 5, Qt.AlignLeft)
        thk_grid.addWidget(QLabel("Start (in)"), 1, 0, Qt.AlignLeft)
        thk_grid.addWidget(_entry(self.thk_start_var, 10), 1, 1, Qt.AlignLeft)
        thk_grid.addWidget(QLabel("Stop"), 1, 2, Qt.AlignLeft)
        thk_grid.addWidget(_entry(self.thk_stop_var, 10), 1, 3, Qt.AlignLeft)
        thk_grid.addWidget(QLabel("Step"), 1, 4, Qt.AlignLeft)
        thk_grid.addWidget(_entry(self.thk_step_var, 10), 1, 5, Qt.AlignLeft)
        thk_grid.addWidget(QLabel("Angle (deg)"), 2, 0, Qt.AlignLeft)
        thk_grid.addWidget(_entry(self.thk_angle_var, 10), 2, 1, Qt.AlignLeft)
        thk_grid.addWidget(QLabel("Wave pol"), 2, 2, Qt.AlignLeft)
        thk_grid.addWidget(
            make_combo(("TE", "TM"), self.thk_wave_pol_var, width=70), 2, 3, Qt.AlignLeft
        )
        thk_grid.setColumnStretch(5, 1)
        thk_layout.addWidget(thk_group)

        (
            thk_unc_group,
            self.thk_unc_details_frame,
            self.thk_unc_t_entry,
            self.thk_unc_eps_entry,
            self.thk_unc_mu_entry,
        ) = _uncertainty_group(
            self.thk_uncertainty_var,
            self.thk_unc_t_pct_var,
            self.thk_unc_eps_pct_var,
            self.thk_unc_mu_pct_var,
            self._sync_thickness_uncertainty_state,
        )
        thk_layout.addWidget(thk_unc_group)
        thk_layout.addWidget(_output_row(self.thk_output_var, self._browse_thickness_output))

        self.thk_compute_btn = QPushButton("Compute")
        self.thk_compute_btn.clicked.connect(self._compute_thickness)
        thk_layout.addWidget(self.thk_compute_btn, 0, Qt.AlignLeft)
        thk_layout.addStretch(1)

        inv_tab = QWidget()
        self.inv_tab = inv_tab
        inv_layout = QVBoxLayout(inv_tab)
        _add_mode("Inverse Design", inv_tab)

        self.inv_freq_target_frame = CollapsibleFrame("Frequency target", expanded=True)
        inv_layout.addWidget(self.inv_freq_target_frame)
        freq_body = QGridLayout(self.inv_freq_target_frame.body)
        freq_body.addWidget(QLabel("Mode"), 0, 0, Qt.AlignLeft)
        freq_mode_combo = make_combo(
            ("Band sweep", "Discrete list"),
            self.inv_freq_mode_var,
            width=130,
            on_change=self._sync_inverse_freq_mode_state,
        )
        freq_body.addWidget(freq_mode_combo, 0, 1, 1, 2, Qt.AlignLeft)
        freq_body.addWidget(QLabel("Band start"), 1, 0, Qt.AlignLeft)
        self.inv_target_start_entry = _entry(self.inv_target_start_var, 8)
        freq_body.addWidget(self.inv_target_start_entry, 1, 1, Qt.AlignLeft)
        freq_body.addWidget(QLabel("Stop"), 1, 2, Qt.AlignLeft)
        self.inv_target_stop_entry = _entry(self.inv_target_stop_var, 8)
        freq_body.addWidget(self.inv_target_stop_entry, 1, 3, Qt.AlignLeft)
        freq_body.addWidget(QLabel("Step"), 1, 4, Qt.AlignLeft)
        self.inv_target_step_entry = _entry(self.inv_target_step_var, 8)
        freq_body.addWidget(self.inv_target_step_entry, 1, 5, Qt.AlignLeft)
        freq_body.addWidget(QLabel("Discrete f (GHz)"), 2, 0, Qt.AlignLeft)
        self.inv_freq_list_entry = _entry(self.inv_freq_list_var)
        freq_body.addWidget(self.inv_freq_list_entry, 2, 1, 1, 5)
        freq_body.setColumnStretch(5, 1)

        self.inv_angle_target_frame = CollapsibleFrame("Angle target", expanded=True)
        inv_layout.addWidget(self.inv_angle_target_frame)
        angle_body = QGridLayout(self.inv_angle_target_frame.body)
        angle_body.addWidget(QLabel("Start (deg)"), 0, 0, Qt.AlignLeft)
        angle_body.addWidget(_entry(self.inv_angle_start_var, 8), 0, 1, Qt.AlignLeft)
        angle_body.addWidget(QLabel("Stop"), 0, 2, Qt.AlignLeft)
        angle_body.addWidget(_entry(self.inv_angle_stop_var, 8), 0, 3, Qt.AlignLeft)
        angle_body.addWidget(QLabel("Step"), 0, 4, Qt.AlignLeft)
        angle_body.addWidget(_entry(self.inv_angle_step_var, 8), 0, 5, Qt.AlignLeft)
        angle_body.addWidget(QLabel("Wave pol"), 0, 6, Qt.AlignLeft)
        angle_body.addWidget(make_combo(("TE", "TM"), self.inv_wave_pol_var, width=60), 0, 7, Qt.AlignLeft)
        angle_body.setColumnStretch(8, 1)

        self.inv_search_frame = CollapsibleFrame("Monte Carlo search", expanded=False)
        inv_layout.addWidget(self.inv_search_frame)
        search_body = QGridLayout(self.inv_search_frame.body)
        search_body.addWidget(QLabel("Samples"), 0, 0, Qt.AlignLeft)
        search_body.addWidget(_entry(self.inv_max_evals_var, 8), 0, 1, Qt.AlignLeft)
        search_body.addWidget(QLabel("Top N"), 0, 2, Qt.AlignLeft)
        search_body.addWidget(_entry(self.inv_top_n_var, 8), 0, 3, Qt.AlignLeft)
        search_body.addWidget(QLabel("Seed"), 0, 4, Qt.AlignLeft)
        search_body.addWidget(_entry(self.inv_seed_var, 10), 0, 5, Qt.AlignLeft)
        refine_check = QCheckBox("Refine top candidates (local search)")
        bind_check_box(self.inv_refine_var, refine_check)
        search_body.addWidget(refine_check, 1, 0, 1, 6, Qt.AlignLeft)
        search_body.setColumnStretch(6, 1)

        self.inv_score_frame = CollapsibleFrame("Robust scoring", expanded=False)
        inv_layout.addWidget(self.inv_score_frame)
        score_body = QGridLayout(self.inv_score_frame.body)
        inv_unc_check = QCheckBox("Enable uncertainty corners")
        bind_check_box(self.inv_uncertainty_var, inv_unc_check)
        inv_unc_check.clicked.connect(self._sync_inverse_uncertainty_state)
        score_body.addWidget(inv_unc_check, 0, 0, 1, 6, Qt.AlignLeft)
        score_body.addWidget(QLabel("T ±%"), 1, 0, Qt.AlignLeft)
        self.inv_unc_t_entry = _entry(self.inv_unc_t_pct_var, 7)
        score_body.addWidget(self.inv_unc_t_entry, 1, 1, Qt.AlignLeft)
        score_body.addWidget(QLabel("Eps ±%"), 1, 2, Qt.AlignLeft)
        self.inv_unc_eps_entry = _entry(self.inv_unc_eps_pct_var, 7)
        score_body.addWidget(self.inv_unc_eps_entry, 1, 3, Qt.AlignLeft)
        score_body.addWidget(QLabel("Mu ±%"), 1, 4, Qt.AlignLeft)
        self.inv_unc_mu_entry = _entry(self.inv_unc_mu_pct_var, 7)
        score_body.addWidget(self.inv_unc_mu_entry, 1, 5, Qt.AlignLeft)
        score_body.addWidget(QLabel("Score"), 2, 0, Qt.AlignLeft)
        score_body.addWidget(
            make_combo(INVERSE_SCORE_MODE_OPTIONS, self.inv_score_mode_var, width=280),
            2,
            1,
            1,
            5,
            Qt.AlignLeft,
        )
        score_body.setColumnStretch(5, 1)

        self.inv_results_frame = CollapsibleFrame("Top candidates", expanded=False)
        inv_layout.addWidget(self.inv_results_frame)
        results_body = QVBoxLayout(self.inv_results_frame.body)
        results_body.setContentsMargins(0, 0, 0, 0)
        self.inv_results_list = QListWidget()
        self.inv_results_list.setMinimumHeight(120)
        self.inv_results_list.itemSelectionChanged.connect(self._update_plot)
        results_body.addWidget(self.inv_results_list)

        inv_actions = QWidget()
        inv_actions_layout = QHBoxLayout(inv_actions)
        inv_actions_layout.setContentsMargins(0, 0, 0, 0)
        self.inv_run_btn = QPushButton("Run Inverse Design")
        self.inv_run_btn.clicked.connect(self._run_inverse_design)
        inv_actions_layout.addWidget(self.inv_run_btn)
        self.inv_apply_btn = QPushButton("Apply Selected")
        self.inv_apply_btn.clicked.connect(self._apply_inverse_candidate)
        inv_actions_layout.addWidget(self.inv_apply_btn)
        inv_actions_layout.addWidget(QLabel("Percentile"))
        self.inv_percentile_entry = _entry(self.inv_percentile_var, 6)
        self.inv_percentile_entry.editingFinished.connect(self._on_inverse_percentile_changed)
        inv_actions_layout.addWidget(self.inv_percentile_entry)
        inv_actions_layout.addWidget(QLabel("%"))
        inv_actions_layout.addStretch(1)
        inv_layout.addWidget(inv_actions)
        inv_layout.addStretch(1)

        # --- Material Mix tab -------------------------------------------------
        mix_tab = QScrollArea()
        mix_tab.setWidgetResizable(True)
        self.mix_tab = mix_tab
        mix_content = QWidget()
        mix_tab.setWidget(mix_content)
        mix_layout = QVBoxLayout(mix_content)
        _add_mode("Material Mix", mix_tab)

        intro = QLabel(
            "Build a volume-based recipe from measured material CSV files. "
            "Predict effective ε/μ for a known recipe or search for recipes "
            "matching target properties or planar-stack performance. Results "
            "remain morphology-model-dependent estimates."
        )
        intro.setWordWrap(True)
        mix_layout.addWidget(intro)

        workflow_frame = QGroupBox("1. Choose task and morphology model")
        workflow_grid = QGridLayout(workflow_frame)
        workflow_grid.addWidget(QLabel("Task"), 0, 0, Qt.AlignLeft)
        workflow_grid.addWidget(
            make_combo(
                MIX_OBJECTIVE_OPTIONS,
                self.mix_objective_var,
                width=310,
                on_change=self._sync_mix_objective_state,
            ),
            0,
            1,
            Qt.AlignLeft,
        )
        workflow_grid.addWidget(QLabel("Effective-medium model"), 1, 0, Qt.AlignLeft)
        workflow_grid.addWidget(
            make_combo(
                MIX_RULE_LABEL_OPTIONS,
                self.mix_rule_var,
                width=410,
                on_change=self._on_mix_model_changed,
            ),
            1,
            1,
            Qt.AlignLeft,
        )
        self.mix_workflow_help_label = QLabel()
        self.mix_workflow_help_label.setWordWrap(True)
        workflow_grid.addWidget(self.mix_workflow_help_label, 2, 0, 1, 2)
        self.mix_model_help_label = QLabel()
        self.mix_model_help_label.setWordWrap(True)
        workflow_grid.addWidget(self.mix_model_help_label, 3, 0, 1, 2)
        workflow_grid.setColumnStretch(1, 1)
        mix_layout.addWidget(workflow_frame)

        mix_comp_frame = CollapsibleFrame(
            "2. Add measured materials and volume recipe", expanded=True
        )
        mix_layout.addWidget(mix_comp_frame)
        comp_body = QVBoxLayout(mix_comp_frame.body)
        comp_body.setContentsMargins(0, 0, 0, 0)
        self.mix_list = QListWidget()
        self.mix_list.setMinimumHeight(130)
        self.mix_list.itemSelectionChanged.connect(self._update_plot)
        self.mix_list.itemDoubleClicked.connect(lambda _item: self._edit_mix_component())
        comp_body.addWidget(self.mix_list)
        comp_btns = QWidget()
        comp_btns_layout = QHBoxLayout(comp_btns)
        comp_btns_layout.setContentsMargins(0, 0, 0, 0)
        self.mix_add_btn = QPushButton("Add measured material…")
        self.mix_add_btn.clicked.connect(self._add_mix_component)
        comp_btns_layout.addWidget(self.mix_add_btn)
        self.mix_edit_btn = QPushButton("Edit selected…")
        self.mix_edit_btn.clicked.connect(self._edit_mix_component)
        comp_btns_layout.addWidget(self.mix_edit_btn)
        self.mix_remove_btn = QPushButton("Remove")
        self.mix_remove_btn.clicked.connect(self._remove_mix_component)
        comp_btns_layout.addWidget(self.mix_remove_btn)
        comp_btns_layout.addStretch(1)
        comp_body.addWidget(comp_btns)

        mix_freq_frame = CollapsibleFrame("3. Choose prediction band", expanded=True)
        mix_layout.addWidget(mix_freq_frame)
        mix_freq_body = QGridLayout(mix_freq_frame.body)
        mix_freq_body.addWidget(QLabel("Mode"), 0, 0, Qt.AlignLeft)
        mix_freq_body.addWidget(
            make_combo(
                ("Band sweep", "Discrete list"),
                self.mix_freq_mode_var,
                width=130,
                on_change=self._sync_mix_freq_mode_state,
            ),
            0,
            1,
            1,
            2,
            Qt.AlignLeft,
        )
        mix_freq_body.addWidget(QLabel("Band start (GHz)"), 1, 0, Qt.AlignLeft)
        self.mix_target_start_entry = _entry(self.mix_target_start_var, 8)
        mix_freq_body.addWidget(self.mix_target_start_entry, 1, 1, Qt.AlignLeft)
        mix_freq_body.addWidget(QLabel("Stop"), 1, 2, Qt.AlignLeft)
        self.mix_target_stop_entry = _entry(self.mix_target_stop_var, 8)
        mix_freq_body.addWidget(self.mix_target_stop_entry, 1, 3, Qt.AlignLeft)
        mix_freq_body.addWidget(QLabel("Step"), 1, 4, Qt.AlignLeft)
        self.mix_target_step_entry = _entry(self.mix_target_step_var, 8)
        mix_freq_body.addWidget(self.mix_target_step_entry, 1, 5, Qt.AlignLeft)
        mix_freq_body.addWidget(QLabel("Discrete frequencies (GHz)"), 2, 0, Qt.AlignLeft)
        self.mix_freq_list_entry = _entry(self.mix_freq_list_var)
        mix_freq_body.addWidget(self.mix_freq_list_entry, 2, 1, 1, 5)
        mix_freq_body.setColumnStretch(5, 1)

        self.mix_prop_frame = CollapsibleFrame(
            "4. Set target effective properties", expanded=True
        )
        mix_layout.addWidget(self.mix_prop_frame)
        mix_prop_body = QGridLayout(self.mix_prop_frame.body)
        mix_prop_body.addWidget(QLabel("Target source"), 0, 0, Qt.AlignLeft)
        mix_prop_body.addWidget(
            make_combo(
                MIX_PROP_SOURCE_OPTIONS,
                self.mix_prop_source_var,
                width=140,
                on_change=self._sync_mix_prop_source_state,
            ),
            0,
            1,
            1,
            2,
            Qt.AlignLeft,
        )
        mix_prop_body.addWidget(QLabel("ε'"), 1, 0, Qt.AlignLeft)
        eps_re_entry = _entry(self.mix_prop_eps_re_var, 8)
        mix_prop_body.addWidget(eps_re_entry, 1, 1, Qt.AlignLeft)
        mix_prop_body.addWidget(QLabel("ε'' (passive loss < 0)"), 1, 2, Qt.AlignLeft)
        eps_im_entry = _entry(self.mix_prop_eps_im_var, 8)
        mix_prop_body.addWidget(eps_im_entry, 1, 3, Qt.AlignLeft)
        mix_prop_body.addWidget(QLabel("μ'"), 1, 4, Qt.AlignLeft)
        mu_re_entry = _entry(self.mix_prop_mu_re_var, 8)
        mix_prop_body.addWidget(mu_re_entry, 1, 5, Qt.AlignLeft)
        mix_prop_body.addWidget(QLabel("μ'' (passive loss < 0)"), 1, 6, Qt.AlignLeft)
        mu_im_entry = _entry(self.mix_prop_mu_im_var, 8)
        mix_prop_body.addWidget(mu_im_entry, 1, 7, Qt.AlignLeft)
        self.mix_prop_const_entries = [eps_re_entry, eps_im_entry, mu_re_entry, mu_im_entry]
        mix_prop_body.addWidget(QLabel("Target material CSV"), 2, 0, Qt.AlignLeft)
        self.mix_prop_file_entry = _entry(self.mix_prop_file_var)
        mix_prop_body.addWidget(self.mix_prop_file_entry, 2, 1, 1, 6)
        self.mix_prop_browse_btn = QPushButton("Browse…")
        self.mix_prop_browse_btn.clicked.connect(self._browse_mix_prop_file)
        mix_prop_body.addWidget(self.mix_prop_browse_btn, 2, 7, Qt.AlignLeft)
        mix_prop_body.addWidget(QLabel("Importance: ε"), 3, 0, Qt.AlignLeft)
        mix_prop_body.addWidget(_entry(self.mix_prop_weps_var, 6), 3, 1, Qt.AlignLeft)
        mix_prop_body.addWidget(QLabel("μ"), 3, 2, Qt.AlignLeft)
        mix_prop_body.addWidget(_entry(self.mix_prop_wmu_var, 6), 3, 3, Qt.AlignLeft)
        mix_prop_body.setColumnStretch(7, 1)

        self.mix_perf_frame = CollapsibleFrame(
            "4. Set target planar-stack performance", expanded=True
        )
        mix_layout.addWidget(self.mix_perf_frame)
        mix_perf_body = QGridLayout(self.mix_perf_frame.body)
        mix_perf_body.addWidget(QLabel("Performance metric"), 0, 0, Qt.AlignLeft)
        mix_perf_body.addWidget(
            make_combo(
                tuple(item[0] for item in MIX_PERFORMANCE_METRIC_OPTIONS),
                self.mix_perf_metric_var,
                width=300,
                on_change=self._on_mix_performance_metric_changed,
            ),
            0,
            1,
            1,
            3,
            Qt.AlignLeft,
        )
        mix_perf_body.addWidget(QLabel("Required threshold"), 1, 0, Qt.AlignLeft)
        mix_perf_body.addWidget(_entry(self.mix_perf_target_var, 9), 1, 1, Qt.AlignLeft)
        self.mix_perf_requirement_label = QLabel()
        mix_perf_body.addWidget(self.mix_perf_requirement_label, 1, 2, 1, 4, Qt.AlignLeft)
        mix_perf_body.addWidget(QLabel("Angle start (deg)"), 2, 0, Qt.AlignLeft)
        mix_perf_body.addWidget(_entry(self.mix_perf_angle_start_var, 8), 2, 1, Qt.AlignLeft)
        mix_perf_body.addWidget(QLabel("Stop"), 2, 2, Qt.AlignLeft)
        mix_perf_body.addWidget(_entry(self.mix_perf_angle_stop_var, 8), 2, 3, Qt.AlignLeft)
        mix_perf_body.addWidget(QLabel("Step"), 2, 4, Qt.AlignLeft)
        mix_perf_body.addWidget(_entry(self.mix_perf_angle_step_var, 8), 2, 5, Qt.AlignLeft)
        mix_perf_body.addWidget(QLabel("Wave polarization"), 3, 0, Qt.AlignLeft)
        mix_perf_body.addWidget(
            make_combo(("TE", "TM"), self.mix_perf_wave_pol_var, width=80),
            3,
            1,
            Qt.AlignLeft,
        )
        perf_help = QLabel(
            "The requirement is enforced at the worst frequency/angle point, "
            "not only on an average. PEC-backed metrics model the mixed layer "
            "on a conductor; air-backed metrics model a free-standing slab."
        )
        perf_help.setWordWrap(True)
        mix_perf_body.addWidget(perf_help, 4, 0, 1, 6)
        mix_perf_body.setColumnStretch(5, 1)

        self.mix_search_frame = CollapsibleFrame(
            "Inverse recipe search settings", expanded=False
        )
        mix_layout.addWidget(self.mix_search_frame)
        mix_search_body = QGridLayout(self.mix_search_frame.body)
        mix_search_body.addWidget(QLabel("Recipe samples"), 0, 0, Qt.AlignLeft)
        mix_search_body.addWidget(_entry(self.mix_max_evals_var, 8), 0, 1, Qt.AlignLeft)
        mix_search_body.addWidget(QLabel("Keep best"), 0, 2, Qt.AlignLeft)
        mix_search_body.addWidget(_entry(self.mix_top_n_var, 8), 0, 3, Qt.AlignLeft)
        mix_search_body.addWidget(QLabel("Seed"), 0, 4, Qt.AlignLeft)
        mix_search_body.addWidget(_entry(self.mix_seed_var, 10), 0, 5, Qt.AlignLeft)
        mix_refine_check = QCheckBox(
            "Refine best recipes on the bounded volume-fraction simplex"
        )
        bind_check_box(self.mix_refine_var, mix_refine_check)
        mix_search_body.addWidget(mix_refine_check, 1, 0, 1, 6, Qt.AlignLeft)
        mix_unc_check = QCheckBox(
            "Include systematic constituent-property tolerance corners"
        )
        bind_check_box(self.mix_uncertainty_var, mix_unc_check)
        mix_unc_check.clicked.connect(self._sync_mix_uncertainty_state)
        mix_search_body.addWidget(mix_unc_check, 2, 0, 1, 6, Qt.AlignLeft)
        mix_search_body.addWidget(QLabel("Layer thickness ±%"), 3, 0, Qt.AlignLeft)
        self.mix_unc_t_entry = _entry(self.mix_unc_t_pct_var, 7)
        mix_search_body.addWidget(self.mix_unc_t_entry, 3, 1, Qt.AlignLeft)
        mix_search_body.addWidget(QLabel("ε constituents ±%"), 3, 2, Qt.AlignLeft)
        self.mix_unc_eps_entry = _entry(self.mix_unc_eps_pct_var, 7)
        mix_search_body.addWidget(self.mix_unc_eps_entry, 3, 3, Qt.AlignLeft)
        mix_search_body.addWidget(QLabel("μ constituents ±%"), 3, 4, Qt.AlignLeft)
        self.mix_unc_mu_entry = _entry(self.mix_unc_mu_pct_var, 7)
        mix_search_body.addWidget(self.mix_unc_mu_entry, 3, 5, Qt.AlignLeft)
        tolerance_help = QLabel(
            "Property tolerance scales all constituent ε values together and "
            "all μ values together (correlated calibration bias). Thickness "
            "tolerance affects performance searches and is ignored for a pure "
            "effective-property target."
        )
        tolerance_help.setWordWrap(True)
        mix_search_body.addWidget(tolerance_help, 4, 0, 1, 6)
        mix_search_body.addWidget(QLabel("Tolerance score"), 5, 0, Qt.AlignLeft)
        mix_search_body.addWidget(
            make_combo(MIX_SCORE_MODE_OPTIONS, self.mix_score_mode_var, width=300),
            5,
            1,
            1,
            5,
            Qt.AlignLeft,
        )
        mix_search_body.setColumnStretch(5, 1)

        self.mix_results_frame = CollapsibleFrame(
            "Results and candidate recipes", expanded=True
        )
        mix_layout.addWidget(self.mix_results_frame)
        mix_results_body = QVBoxLayout(self.mix_results_frame.body)
        mix_results_body.setContentsMargins(0, 0, 0, 0)
        self.mix_results_list = QListWidget()
        self.mix_results_list.setMinimumHeight(110)
        self.mix_results_list.itemSelectionChanged.connect(self._update_plot)
        mix_results_body.addWidget(self.mix_results_list)
        self.mix_summary_label = QLabel(
            "Add at least two measured materials, then calculate a known "
            "recipe or find recipes for a target."
        )
        self.mix_summary_label.setWordWrap(True)
        mix_results_body.addWidget(self.mix_summary_label)

        mix_actions = QWidget()
        mix_actions_layout = QHBoxLayout(mix_actions)
        mix_actions_layout.setContentsMargins(0, 0, 0, 0)
        self.mix_preview_btn = QPushButton("Calculate recipe")
        self.mix_preview_btn.clicked.connect(self._preview_mix)
        mix_actions_layout.addWidget(self.mix_preview_btn)
        self.mix_run_btn = QPushButton("Find matching recipes")
        self.mix_run_btn.clicked.connect(self._run_mix_design)
        mix_actions_layout.addWidget(self.mix_run_btn)
        mix_actions_layout.addStretch(1)
        mix_actions_layout.addWidget(QLabel("Stack-layer thickness (in)"))
        mix_actions_layout.addWidget(_entry(self.mix_thickness_var, 7))
        self.mix_apply_btn = QPushButton("Add selected as layer…")
        self.mix_apply_btn.clicked.connect(self._apply_mix_as_layer)
        mix_actions_layout.addWidget(self.mix_apply_btn)
        self.mix_export_btn = QPushButton("Export selected CSV…")
        self.mix_export_btn.clicked.connect(self._export_mix_material)
        mix_actions_layout.addWidget(self.mix_export_btn)
        mix_layout.addWidget(mix_actions)
        self._bind_mix_input_invalidation()
        mix_layout.addStretch(1)

        layers_group = QGroupBox("Layers (top to bottom)")
        layers_layout = QHBoxLayout(layers_group)
        self.layer_list = QListWidget()
        self.layer_list.setMinimumHeight(200)
        layers_layout.addWidget(self.layer_list, 1)

        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(QLabel("Visual stack (real-time)"))
        self.layer_preview = LayerPreview(self._draw_layer_preview)
        preview_layout.addWidget(self.layer_preview, 1)
        layers_layout.addWidget(preview_container, 1)

        btns = QWidget()
        btns_layout = QVBoxLayout(btns)
        btns_layout.setContentsMargins(0, 0, 0, 0)
        self.layer_add_btn = QPushButton("Add Layer")
        self.layer_add_btn.clicked.connect(self._add_layer)
        btns_layout.addWidget(self.layer_add_btn)
        self.layer_add_sheet_btn = QPushButton("Add Sheet")
        self.layer_add_sheet_btn.clicked.connect(self._add_sheet)
        btns_layout.addWidget(self.layer_add_sheet_btn)
        self.layer_edit_btn = QPushButton("Edit")
        self.layer_edit_btn.clicked.connect(self._edit_layer)
        btns_layout.addWidget(self.layer_edit_btn)
        self.layer_remove_btn = QPushButton("Remove")
        self.layer_remove_btn.clicked.connect(self._remove_layer)
        btns_layout.addWidget(self.layer_remove_btn)
        self.layer_up_btn = QPushButton("Move Up")
        self.layer_up_btn.clicked.connect(self._move_up)
        btns_layout.addWidget(self.layer_up_btn)
        self.layer_down_btn = QPushButton("Move Down")
        self.layer_down_btn.clicked.connect(self._move_down)
        btns_layout.addWidget(self.layer_down_btn)
        btns_layout.addStretch(1)
        layers_layout.addWidget(btns)
        # Finish the navigation rail and wire mode switching.
        nav_layout.addStretch(1)
        self.nav_group.idClicked.connect(self._select_mode)

        # Vertical workspace splitter: parameter inputs and the material stack
        # share the top band; the visualization spans the full width below.
        work_split = QSplitter(Qt.Vertical)
        root_layout.addWidget(work_split, 1)

        top_pane = QWidget()
        top_layout = QHBoxLayout(top_pane)
        top_layout.setContentsMargins(12, 12, 12, 6)
        top_layout.addWidget(self.mode_stack, 3)
        top_layout.addWidget(layers_group, 2)
        work_split.addWidget(top_pane)

        bottom_pane = QWidget()
        right_layout = QVBoxLayout(bottom_pane)
        right_layout.setContentsMargins(12, 6, 12, 12)
        work_split.addWidget(bottom_pane)

        plot_opts = QGroupBox("Heatmap Controls")
        opts_grid = QGridLayout(plot_opts)
        opts_grid.addWidget(QLabel("Metric"), 0, 0, Qt.AlignLeft)
        metric_combo = make_combo(
            [label for label, _key in HEATMAP_METRIC_OPTIONS],
            self.heatmap_metric_var,
            on_change=self._update_plot,
        )
        opts_grid.addWidget(metric_combo, 0, 1)
        opts_grid.addWidget(QLabel("Uncertainty view"), 0, 2, Qt.AlignLeft)
        unc_view_combo = make_combo(
            [label for label, _key in UNCERTAINTY_VIEW_OPTIONS],
            self.uncertainty_view_var,
            width=130,
            on_change=self._update_plot,
        )
        opts_grid.addWidget(unc_view_combo, 0, 3)
        update_btn = QPushButton("Update Plot")
        update_btn.clicked.connect(self._update_plot)
        opts_grid.addWidget(update_btn, 0, 4)
        self.slice_x_label = QLabel("Angle slice (deg)")
        opts_grid.addWidget(self.slice_x_label, 1, 0, Qt.AlignLeft)
        self.slice_angle_entry = _entry(self.slice_angle_var, 10)
        self.slice_angle_entry.returnPressed.connect(self._apply_manual_slices)
        opts_grid.addWidget(self.slice_angle_entry, 1, 1, Qt.AlignLeft)
        opts_grid.addWidget(QLabel("Freq slice (GHz)"), 1, 2, Qt.AlignLeft)
        self.slice_freq_entry = _entry(self.slice_freq_var, 10)
        self.slice_freq_entry.returnPressed.connect(self._apply_manual_slices)
        opts_grid.addWidget(self.slice_freq_entry, 1, 3, Qt.AlignLeft)
        cbar_auto_check = QCheckBox("Auto color scale")
        bind_check_box(self.cbar_auto_var, cbar_auto_check)
        cbar_auto_check.clicked.connect(self._sync_cbar_state)
        opts_grid.addWidget(cbar_auto_check, 1, 4, Qt.AlignLeft)
        opts_grid.addWidget(QLabel("Min"), 1, 5, Qt.AlignRight)
        self.cbar_min_entry = _entry(self.cbar_min_var, 10)
        self.cbar_min_entry.returnPressed.connect(self._update_plot)
        opts_grid.addWidget(self.cbar_min_entry, 1, 6, Qt.AlignLeft)
        opts_grid.addWidget(QLabel("Max"), 1, 7, Qt.AlignRight)
        self.cbar_max_entry = _entry(self.cbar_max_var, 10)
        self.cbar_max_entry.returnPressed.connect(self._update_plot)
        opts_grid.addWidget(self.cbar_max_entry, 1, 8, Qt.AlignLeft)
        save_plot_btn = QPushButton("Save Plot")
        save_plot_btn.clicked.connect(self._save_plot)
        opts_grid.addWidget(save_plot_btn, 1, 9)
        save_heatmap_btn = QPushButton("Save Heatmap Only")
        save_heatmap_btn.clicked.connect(self._save_heatmap_only)
        opts_grid.addWidget(save_heatmap_btn, 1, 10)
        opts_grid.setColumnStretch(1, 1)
        opts_grid.setColumnStretch(3, 1)
        right_layout.addWidget(plot_opts)
        self._sync_cbar_state()

        self.plot_frame = QGroupBox("Heatmap")
        plot_frame_layout = QVBoxLayout(self.plot_frame)
        plot_frame_layout.setContentsMargins(4, 4, 4, 4)
        if MPL_AVAILABLE:
            self.fig = Figure(figsize=(8.0, 4.6), dpi=100)
            # Heatmap occupies the full-height left column; the frequency and
            # angle slice plots stack in the right column (two rows).
            gs = self.fig.add_gridspec(
                2, 2, width_ratios=[2.3, 1.0], wspace=0.5, hspace=0.85
            )
            self.ax_heatmap = self.fig.add_subplot(gs[:, 0])
            self.ax_freq_slice = self.fig.add_subplot(gs[0, 1])
            self.ax_angle_slice = self.fig.add_subplot(gs[1, 1])
            self.canvas = FigureCanvas(self.fig)
            self.heatmap_cbar = None
            self.heatmap_click_cid = self.canvas.mpl_connect("button_press_event", self._on_plot_click)
            plot_frame_layout.addWidget(self.canvas)
            self._draw_plot_placeholder("Run the Off Angle compute to populate plot.")
        else:
            self.fig = None
            self.ax_heatmap = None
            self.ax_freq_slice = None
            self.ax_angle_slice = None
            self.canvas = None
            self.heatmap_cbar = None
            plot_frame_layout.addWidget(
                QLabel("Matplotlib not available. Install matplotlib to enable plotting.")
            )
        right_layout.addWidget(self.plot_frame, 1)

        work_split.setStretchFactor(0, 0)
        work_split.setStretchFactor(1, 1)
        work_split.setSizes([320, 380])

        # Status text and the busy indicator live in the window status bar.
        self.status_label = QLabel(self.status_var.get())
        self.status_var.valueChanged.connect(self.status_label.setText)
        self.statusBar().addWidget(self.status_label)
        self.status_progress = QProgressBar()
        self.status_progress.setRange(0, 0)
        self.status_progress.setMaximumWidth(120)
        self.status_progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.status_progress)

        self._sync_uncertainty_state()
        self._sync_angle_uncertainty_state()
        self._sync_thickness_uncertainty_state()
        self._sync_inverse_freq_mode_state()
        self._sync_inverse_uncertainty_state()
        self._sync_mix_freq_mode_state()
        self._sync_mix_uncertainty_state()
        self._sync_mix_objective_state()
        self._refresh_mix_components_list()
        self._refresh_thickness_layers()

    def _on_theme_toggle(self) -> None:
        self._apply_theme()

    def _theme_colors(self) -> dict[str, object]:
        return DARK_THEME if self.dark_mode_var.get() else LIGHT_THEME

    def _style_plot_axis(self, axis: object) -> None:
        style_axis(axis, self._colors)

    def _apply_theme(self) -> None:
        colors = self._theme_colors()
        self._colors = colors

        qss = f"""
        QWidget {{ background-color: {colors['window_bg']}; color: {colors['text']}; }}
        QGroupBox {{
            background-color: {colors['panel_bg']};
            border: 1px solid {colors['preview_border']};
            border-radius: 4px;
            margin-top: 8px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 3px;
            color: {colors['text']};
        }}
        QLabel {{ background: transparent; color: {colors['text']}; }}
        QLabel:disabled {{ color: {colors['field_disabled_fg']}; }}
        QToolButton#CollapsibleHeader {{
            border: none;
            text-align: left;
            padding: 4px 6px;
            background: transparent;
            color: {colors['text']};
        }}
        QLineEdit {{
            background-color: {colors['field_bg']};
            color: {colors['field_fg']};
            border: 1px solid {colors['preview_border']};
            border-radius: 3px;
            padding: 2px 4px;
        }}
        QLineEdit:disabled {{
            background-color: {colors['field_disabled_bg']};
            color: {colors['field_disabled_fg']};
        }}
        QPushButton {{
            background-color: {colors['button_bg']};
            color: {colors['text']};
            border: 1px solid {colors['preview_border']};
            border-radius: 3px;
            padding: 4px 10px;
        }}
        QPushButton:hover {{ background-color: {colors['button_active_bg']}; }}
        QPushButton:disabled {{
            background-color: {colors['field_disabled_bg']};
            color: {colors['field_disabled_fg']};
        }}
        QCheckBox {{ background: transparent; color: {colors['text']}; }}
        QCheckBox:disabled {{ color: {colors['field_disabled_fg']}; }}
        QComboBox {{
            background-color: {colors['field_bg']};
            color: {colors['field_fg']};
            border: 1px solid {colors['preview_border']};
            border-radius: 3px;
            padding: 2px 4px;
        }}
        QComboBox:disabled {{
            background-color: {colors['field_disabled_bg']};
            color: {colors['field_disabled_fg']};
        }}
        QComboBox QAbstractItemView {{
            background-color: {colors['field_bg']};
            color: {colors['field_fg']};
            selection-background-color: {colors['selection_bg']};
            selection-color: {colors['selection_fg']};
        }}
        QListWidget {{
            background-color: {colors['field_bg']};
            color: {colors['field_fg']};
            border: 1px solid {colors['preview_border']};
        }}
        QListWidget::item:selected {{
            background-color: {colors['selection_bg']};
            color: {colors['selection_fg']};
        }}
        QTabWidget::pane {{ border: 1px solid {colors['preview_border']}; }}
        QTabBar::tab {{
            background: {colors['button_bg']};
            color: {colors['text']};
            padding: 5px 10px;
        }}
        QTabBar::tab:selected {{ background: {colors['head_bg']}; color: {colors['field_fg']}; }}
        QTabBar::tab:hover {{ background: {colors['button_active_bg']}; }}
        QProgressBar {{
            background-color: {colors['field_disabled_bg']};
            border: none;
            border-radius: 3px;
        }}
        QProgressBar::chunk {{ background-color: {colors['accent']}; }}
        QSplitter::handle {{ background-color: {colors['preview_border']}; }}
        QMenuBar {{ background-color: {colors['panel_bg']}; color: {colors['text']}; }}
        QMenuBar::item {{ background: transparent; padding: 4px 10px; }}
        QMenuBar::item:selected {{ background-color: {colors['button_active_bg']}; }}
        QMenu {{
            background-color: {colors['field_bg']};
            color: {colors['field_fg']};
            border: 1px solid {colors['preview_border']};
        }}
        QMenu::item:selected {{
            background-color: {colors['selection_bg']};
            color: {colors['selection_fg']};
        }}
        QStatusBar {{ background-color: {colors['panel_bg']}; color: {colors['muted_text']}; }}
        QStatusBar QLabel {{ color: {colors['muted_text']}; }}
        QFrame#NavRail {{
            background-color: {colors['field_bg']};
            border-right: 1px solid {colors['preview_border']};
        }}
        QLabel#NavBrand {{
            color: {colors['accent']};
            font-weight: 600;
            font-size: 15px;
            padding: 2px 2px;
        }}
        QToolButton#ModeNavButton {{
            border: none;
            border-radius: 5px;
            text-align: left;
            padding: 9px 12px;
            color: {colors['text']};
            background: transparent;
        }}
        QToolButton#ModeNavButton:hover {{ background-color: {colors['button_active_bg']}; }}
        QToolButton#ModeNavButton:checked {{
            background-color: {colors['selection_bg']};
            color: {colors['selection_fg']};
        }}
        """
        self.setStyleSheet(qss)
        self.layer_preview.update()

        if self.canvas is not None and self.fig is not None:
            self.fig.patch.set_facecolor(colors["plot_bg"])
            self._update_plot()

    def _browse_output(self) -> None:
        p = filedialog.asksaveasfilename(
            title="Select output file",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )
        if p:
            self.output_var.set(p)

    def _browse_ibc_batch_output_dir(self) -> None:
        current = self.ibc_batch_output_dir_var.get().strip()
        p = filedialog.askdirectory(
            parent=self,
            title="Select IBC batch output folder",
            initialdir=current if current and Path(current).is_dir() else "",
        )
        if p:
            self.ibc_batch_output_dir_var.set(p)

    def _browse_angle_output(self) -> None:
        p = filedialog.asksaveasfilename(
            title="Select output file",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )
        if p:
            self.angle_output_var.set(p)

    def _browse_thickness_output(self) -> None:
        p = filedialog.asksaveasfilename(
            title="Select output file",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )
        if p:
            self.thk_output_var.set(p)

    def _coerce_bool(self, value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return False

    def _collect_project_state(self) -> dict[str, object]:
        controls: dict[str, object] = {
            "f_start": self.f_start_var.get(),
            "f_stop": self.f_stop_var.get(),
            "f_step": self.f_step_var.get(),
            "backing": self.backing_var.get(),
            "output": self.output_var.get(),
            "ibc_batch_layer": self.ibc_batch_layer_var.get(),
            "ibc_batch_start": self.ibc_batch_start_var.get(),
            "ibc_batch_stop": self.ibc_batch_stop_var.get(),
            "ibc_batch_step": self.ibc_batch_step_var.get(),
            "ibc_batch_unit": self.ibc_batch_unit_var.get(),
            "ibc_batch_output_dir": self.ibc_batch_output_dir_var.get(),
            "ibc_batch_prefix": self.ibc_batch_prefix_var.get(),
            "uncertainty": self.uncertainty_var.get(),
            "unc_t_pct": self.unc_t_pct_var.get(),
            "unc_eps_pct": self.unc_eps_pct_var.get(),
            "unc_mu_pct": self.unc_mu_pct_var.get(),
            "angle_f_start": self.angle_f_start_var.get(),
            "angle_f_stop": self.angle_f_stop_var.get(),
            "angle_f_step": self.angle_f_step_var.get(),
            "angle_start": self.angle_start_var.get(),
            "angle_stop": self.angle_stop_var.get(),
            "angle_step": self.angle_step_var.get(),
            "wave_pol": self.wave_pol_var.get(),
            "angle_output": self.angle_output_var.get(),
            "angle_uncertainty": self.angle_uncertainty_var.get(),
            "angle_unc_t_pct": self.angle_unc_t_pct_var.get(),
            "angle_unc_eps_pct": self.angle_unc_eps_pct_var.get(),
            "angle_unc_mu_pct": self.angle_unc_mu_pct_var.get(),
            "thk_f_start": self.thk_f_start_var.get(),
            "thk_f_stop": self.thk_f_stop_var.get(),
            "thk_f_step": self.thk_f_step_var.get(),
            "thk_start": self.thk_start_var.get(),
            "thk_stop": self.thk_stop_var.get(),
            "thk_step": self.thk_step_var.get(),
            "thk_layer": self.thk_layer_var.get(),
            "thk_angle": self.thk_angle_var.get(),
            "thk_wave_pol": self.thk_wave_pol_var.get(),
            "thk_output": self.thk_output_var.get(),
            "thk_uncertainty": self.thk_uncertainty_var.get(),
            "thk_unc_t_pct": self.thk_unc_t_pct_var.get(),
            "thk_unc_eps_pct": self.thk_unc_eps_pct_var.get(),
            "thk_unc_mu_pct": self.thk_unc_mu_pct_var.get(),
            "heatmap_metric": self.heatmap_metric_var.get(),
            "uncertainty_view": self.uncertainty_view_var.get(),
            "cbar_auto": self.cbar_auto_var.get(),
            "cbar_min": self.cbar_min_var.get(),
            "cbar_max": self.cbar_max_var.get(),
            "slice_angle": self.slice_angle_var.get(),
            "slice_freq": self.slice_freq_var.get(),
            "inv_freq_mode": self.inv_freq_mode_var.get(),
            "inv_freq_list": self.inv_freq_list_var.get(),
            "inv_target_start": self.inv_target_start_var.get(),
            "inv_target_stop": self.inv_target_stop_var.get(),
            "inv_target_step": self.inv_target_step_var.get(),
            "inv_angle_start": self.inv_angle_start_var.get(),
            "inv_angle_stop": self.inv_angle_stop_var.get(),
            "inv_angle_step": self.inv_angle_step_var.get(),
            "inv_wave_pol": self.inv_wave_pol_var.get(),
            "inv_max_evals": self.inv_max_evals_var.get(),
            "inv_top_n": self.inv_top_n_var.get(),
            "inv_percentile": self.inv_percentile_var.get(),
            "inv_uncertainty": self.inv_uncertainty_var.get(),
            "inv_unc_t_pct": self.inv_unc_t_pct_var.get(),
            "inv_unc_eps_pct": self.inv_unc_eps_pct_var.get(),
            "inv_unc_mu_pct": self.inv_unc_mu_pct_var.get(),
            "inv_score_mode": self.inv_score_mode_var.get(),
            "inv_seed": self.inv_seed_var.get(),
            "inv_refine": self.inv_refine_var.get(),
            "mix_rule": self.mix_rule_var.get(),
            "mix_objective": self.mix_objective_var.get(),
            "mix_prop_source": self.mix_prop_source_var.get(),
            "mix_prop_eps_re": self.mix_prop_eps_re_var.get(),
            "mix_prop_eps_im": self.mix_prop_eps_im_var.get(),
            "mix_prop_mu_re": self.mix_prop_mu_re_var.get(),
            "mix_prop_mu_im": self.mix_prop_mu_im_var.get(),
            "mix_prop_file": self.mix_prop_file_var.get(),
            "mix_prop_weps": self.mix_prop_weps_var.get(),
            "mix_prop_wmu": self.mix_prop_wmu_var.get(),
            "mix_perf_metric": self.mix_perf_metric_var.get(),
            "mix_perf_target": self.mix_perf_target_var.get(),
            "mix_perf_angle_start": self.mix_perf_angle_start_var.get(),
            "mix_perf_angle_stop": self.mix_perf_angle_stop_var.get(),
            "mix_perf_angle_step": self.mix_perf_angle_step_var.get(),
            "mix_perf_wave_pol": self.mix_perf_wave_pol_var.get(),
            "mix_thickness": self.mix_thickness_var.get(),
            "mix_freq_mode": self.mix_freq_mode_var.get(),
            "mix_freq_list": self.mix_freq_list_var.get(),
            "mix_target_start": self.mix_target_start_var.get(),
            "mix_target_stop": self.mix_target_stop_var.get(),
            "mix_target_step": self.mix_target_step_var.get(),
            "mix_max_evals": self.mix_max_evals_var.get(),
            "mix_top_n": self.mix_top_n_var.get(),
            "mix_seed": self.mix_seed_var.get(),
            "mix_score_mode": self.mix_score_mode_var.get(),
            "mix_unc_t_pct": self.mix_unc_t_pct_var.get(),
            "mix_unc_eps_pct": self.mix_unc_eps_pct_var.get(),
            "mix_unc_mu_pct": self.mix_unc_mu_pct_var.get(),
            "mix_refine": self.mix_refine_var.get(),
            "mix_uncertainty": self.mix_uncertainty_var.get(),
            "dark_mode": self.dark_mode_var.get(),
        }
        return {
            "layers": [layer_config_to_dict(layer) for layer in self.layers],
            "controls": controls,
            "mixes": {"components": [dict(c) for c in self.mix_components]},
        }

    def _apply_project_state(self, state: dict[str, object]) -> None:
        if not isinstance(state, dict):
            raise ValueError("Project state must be an object.")
        layers_data = state.get("layers", [])
        controls = state.get("controls", {})
        if not isinstance(layers_data, list):
            raise ValueError("Project layers must be a list.")
        if not isinstance(controls, dict):
            raise ValueError("Project controls must be an object.")

        loaded_layers: list[LayerConfig] = []
        for idx, raw_layer in enumerate(layers_data, start=1):
            if not isinstance(raw_layer, dict):
                raise ValueError(f"Layer {idx}: expected an object.")
            loaded_layers.append(layer_config_from_dict(raw_layer, idx))

        str_vars: dict[str, StringVar] = {
            "f_start": self.f_start_var,
            "f_stop": self.f_stop_var,
            "f_step": self.f_step_var,
            "backing": self.backing_var,
            "output": self.output_var,
            "ibc_batch_start": self.ibc_batch_start_var,
            "ibc_batch_stop": self.ibc_batch_stop_var,
            "ibc_batch_step": self.ibc_batch_step_var,
            "ibc_batch_unit": self.ibc_batch_unit_var,
            "ibc_batch_output_dir": self.ibc_batch_output_dir_var,
            "ibc_batch_prefix": self.ibc_batch_prefix_var,
            "unc_t_pct": self.unc_t_pct_var,
            "unc_eps_pct": self.unc_eps_pct_var,
            "unc_mu_pct": self.unc_mu_pct_var,
            "angle_f_start": self.angle_f_start_var,
            "angle_f_stop": self.angle_f_stop_var,
            "angle_f_step": self.angle_f_step_var,
            "angle_start": self.angle_start_var,
            "angle_stop": self.angle_stop_var,
            "angle_step": self.angle_step_var,
            "wave_pol": self.wave_pol_var,
            "angle_output": self.angle_output_var,
            "angle_unc_t_pct": self.angle_unc_t_pct_var,
            "angle_unc_eps_pct": self.angle_unc_eps_pct_var,
            "angle_unc_mu_pct": self.angle_unc_mu_pct_var,
            "thk_f_start": self.thk_f_start_var,
            "thk_f_stop": self.thk_f_stop_var,
            "thk_f_step": self.thk_f_step_var,
            "thk_start": self.thk_start_var,
            "thk_stop": self.thk_stop_var,
            "thk_step": self.thk_step_var,
            "thk_angle": self.thk_angle_var,
            "thk_wave_pol": self.thk_wave_pol_var,
            "thk_output": self.thk_output_var,
            "thk_unc_t_pct": self.thk_unc_t_pct_var,
            "thk_unc_eps_pct": self.thk_unc_eps_pct_var,
            "thk_unc_mu_pct": self.thk_unc_mu_pct_var,
            "heatmap_metric": self.heatmap_metric_var,
            "uncertainty_view": self.uncertainty_view_var,
            "cbar_min": self.cbar_min_var,
            "cbar_max": self.cbar_max_var,
            "slice_angle": self.slice_angle_var,
            "slice_freq": self.slice_freq_var,
            "inv_freq_mode": self.inv_freq_mode_var,
            "inv_freq_list": self.inv_freq_list_var,
            "inv_target_start": self.inv_target_start_var,
            "inv_target_stop": self.inv_target_stop_var,
            "inv_target_step": self.inv_target_step_var,
            "inv_angle_start": self.inv_angle_start_var,
            "inv_angle_stop": self.inv_angle_stop_var,
            "inv_angle_step": self.inv_angle_step_var,
            "inv_wave_pol": self.inv_wave_pol_var,
            "inv_max_evals": self.inv_max_evals_var,
            "inv_top_n": self.inv_top_n_var,
            "inv_percentile": self.inv_percentile_var,
            "inv_unc_t_pct": self.inv_unc_t_pct_var,
            "inv_unc_eps_pct": self.inv_unc_eps_pct_var,
            "inv_unc_mu_pct": self.inv_unc_mu_pct_var,
            "inv_score_mode": self.inv_score_mode_var,
            "inv_seed": self.inv_seed_var,
            "mix_rule": self.mix_rule_var,
            "mix_objective": self.mix_objective_var,
            "mix_prop_source": self.mix_prop_source_var,
            "mix_prop_eps_re": self.mix_prop_eps_re_var,
            "mix_prop_eps_im": self.mix_prop_eps_im_var,
            "mix_prop_mu_re": self.mix_prop_mu_re_var,
            "mix_prop_mu_im": self.mix_prop_mu_im_var,
            "mix_prop_file": self.mix_prop_file_var,
            "mix_prop_weps": self.mix_prop_weps_var,
            "mix_prop_wmu": self.mix_prop_wmu_var,
            "mix_perf_metric": self.mix_perf_metric_var,
            "mix_perf_target": self.mix_perf_target_var,
            "mix_perf_angle_start": self.mix_perf_angle_start_var,
            "mix_perf_angle_stop": self.mix_perf_angle_stop_var,
            "mix_perf_angle_step": self.mix_perf_angle_step_var,
            "mix_perf_wave_pol": self.mix_perf_wave_pol_var,
            "mix_thickness": self.mix_thickness_var,
            "mix_freq_mode": self.mix_freq_mode_var,
            "mix_freq_list": self.mix_freq_list_var,
            "mix_target_start": self.mix_target_start_var,
            "mix_target_stop": self.mix_target_stop_var,
            "mix_target_step": self.mix_target_step_var,
            "mix_max_evals": self.mix_max_evals_var,
            "mix_top_n": self.mix_top_n_var,
            "mix_seed": self.mix_seed_var,
            "mix_score_mode": self.mix_score_mode_var,
            "mix_unc_t_pct": self.mix_unc_t_pct_var,
            "mix_unc_eps_pct": self.mix_unc_eps_pct_var,
            "mix_unc_mu_pct": self.mix_unc_mu_pct_var,
        }
        bool_vars: dict[str, BooleanVar] = {
            "uncertainty": self.uncertainty_var,
            "angle_uncertainty": self.angle_uncertainty_var,
            "thk_uncertainty": self.thk_uncertainty_var,
            "cbar_auto": self.cbar_auto_var,
            "inv_uncertainty": self.inv_uncertainty_var,
            "inv_refine": self.inv_refine_var,
            "mix_refine": self.mix_refine_var,
            "mix_uncertainty": self.mix_uncertainty_var,
            "dark_mode": self.dark_mode_var,
        }

        for key, var in str_vars.items():
            if key in controls:
                value = str(controls[key])
                if key in {
                    "wave_pol",
                    "thk_wave_pol",
                    "inv_wave_pol",
                    "mix_perf_wave_pol",
                }:
                    # Migrate legacy HH/VV project values to unambiguous
                    # plane-wave TE/TM labels.
                    value = normalize_wave_polarization(value).upper()
                elif key == "backing":
                    value = normalize_backing(value)
                elif key == "mix_objective":
                    lowered = value.strip().lower()
                    if "performance" in lowered:
                        value = MIX_OBJECTIVE_PERFORMANCE
                    elif lowered.startswith("match") or lowered.startswith("find"):
                        value = MIX_OBJECTIVE_PROPERTY
                    elif lowered.startswith("predict"):
                        value = MIX_OBJECTIVE_FORWARD
                var.set(value)
        for key, var in bool_vars.items():
            if key in controls:
                var.set(self._coerce_bool(controls[key]))

        mixes = state.get("mixes", {})
        mix_components: list[dict] = []
        if isinstance(mixes, dict):
            raw_components = mixes.get("components", [])
            if isinstance(raw_components, list):
                for raw in raw_components:
                    if isinstance(raw, dict):
                        try:
                            migrated = dict(raw)
                            if migrated.get("units") != "volume_percent":
                                # Old projects stored ambiguous relative-parts
                                # search ranges. Preserve the recipe ratio but
                                # reset inverse bounds to explicit 0..100 vol%.
                                migrated["min"] = 0.0
                                migrated["max"] = 100.0
                                migrated["units"] = "volume_percent"
                            mix_components.append(self._coerce_mix_component(migrated))
                        except Exception:
                            continue

        self.layers = loaded_layers
        self.mix_components = mix_components
        self._refresh_layers()
        # The Thickness layer choices are rebuilt from the restored stack, so
        # the saved selection is re-applied after the combo is repopulated.
        if "thk_layer" in controls:
            saved_layer = str(controls["thk_layer"])
            if saved_layer in [label for _idx, label in self._thickness_layer_choices()]:
                self.thk_layer_var.set(saved_layer)
        if "ibc_batch_layer" in controls:
            saved_layer = str(controls["ibc_batch_layer"])
            if saved_layer in [label for _idx, label in self._thickness_layer_choices()]:
                self.ibc_batch_layer_var.set(saved_layer)
        self._refresh_ibc_batch_preview()
        self._sync_uncertainty_state()
        self._sync_angle_uncertainty_state()
        self._sync_thickness_uncertainty_state()
        self._sync_inverse_freq_mode_state()
        self._sync_inverse_uncertainty_state()
        self._sync_mix_freq_mode_state()
        self._sync_mix_uncertainty_state()
        self._sync_mix_objective_state()
        self._sync_cbar_state()

        self._apply_theme()
        self.last_heatmap_results = None
        self.last_heatmap_uncertainty_min = None
        self.last_heatmap_uncertainty_max = None
        self.last_thickness_results = None
        self.last_thickness_uncertainty_min = None
        self.last_thickness_uncertainty_max = None
        self.inverse_plot_freqs = []
        self.inverse_plot_samples = []
        self.selected_x_idx = None
        self.selected_freq_idx = None
        self.inverse_candidates = []
        self.mix_candidates = []
        self.mix_plot_data = []
        self.mix_preview = None
        self._refresh_inverse_results_list()
        self._refresh_mix_components_list()
        self._refresh_mix_results_list()
        self._update_plot()

    def _save_project(self) -> None:
        try:
            if self.project_path is None:
                path_str = filedialog.asksaveasfilename(
                    title="Save project",
                    defaultextension=".json",
                    filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
                )
                if not path_str:
                    return
                self.project_path = Path(path_str)
            save_project_file(self.project_path, self._collect_project_state())
            messagebox.showinfo("Project", f"Saved project to:\n{self.project_path}")
        except Exception as exc:
            messagebox.showerror("Project Save Error", str(exc))

    def _load_project(self) -> None:
        try:
            path_str = filedialog.askopenfilename(
                title="Load project",
                filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            )
            if not path_str:
                return
            path = Path(path_str)
            state = load_project_file(path)
            self._apply_project_state(state)
            self.project_path = path
            messagebox.showinfo("Project", f"Loaded project from:\n{path}")
        except Exception as exc:
            messagebox.showerror("Project Load Error", str(exc))

    def _sync_cbar_state(self) -> None:
        enabled = not self.cbar_auto_var.get()
        self.cbar_min_entry.setEnabled(enabled)
        self.cbar_max_entry.setEnabled(enabled)
        if self.canvas is not None:
            self._update_plot()

    def _sync_uncertainty_state(self) -> None:
        enabled = self.uncertainty_var.get()
        self.unc_details_frame.setVisible(enabled)
        self.unc_t_entry.setEnabled(enabled)
        self.unc_eps_entry.setEnabled(enabled)
        self.unc_mu_entry.setEnabled(enabled)

    def _sync_angle_uncertainty_state(self) -> None:
        enabled = self.angle_uncertainty_var.get()
        self.angle_unc_details_frame.setVisible(enabled)
        self.angle_unc_t_entry.setEnabled(enabled)
        self.angle_unc_eps_entry.setEnabled(enabled)
        self.angle_unc_mu_entry.setEnabled(enabled)

    def _sync_thickness_uncertainty_state(self) -> None:
        enabled = self.thk_uncertainty_var.get()
        if self.thk_unc_details_frame is None:
            return
        self.thk_unc_details_frame.setVisible(enabled)
        self.thk_unc_t_entry.setEnabled(enabled)
        self.thk_unc_eps_entry.setEnabled(enabled)
        self.thk_unc_mu_entry.setEnabled(enabled)

    def _sync_inverse_uncertainty_state(self) -> None:
        enabled = self.inv_uncertainty_var.get()
        if self.inv_unc_t_entry is not None:
            self.inv_unc_t_entry.setEnabled(enabled)
        if self.inv_unc_eps_entry is not None:
            self.inv_unc_eps_entry.setEnabled(enabled)
        if self.inv_unc_mu_entry is not None:
            self.inv_unc_mu_entry.setEnabled(enabled)

    def _sync_inverse_freq_mode_state(self) -> None:
        mode = self.inv_freq_mode_var.get().strip().lower()
        band_enabled = mode.startswith("band")
        for entry in (
            self.inv_target_start_entry,
            self.inv_target_stop_entry,
            self.inv_target_step_entry,
        ):
            if entry is not None:
                entry.setEnabled(band_enabled)
        if self.inv_freq_list_entry is not None:
            self.inv_freq_list_entry.setEnabled(not band_enabled)

    def _on_inverse_percentile_changed(self) -> None:
        text = self.inv_percentile_var.get().strip()
        if not text:
            self.inv_percentile_var.set("10")
            self._update_plot()
            return
        try:
            p = float(text)
        except Exception:
            messagebox.showerror("Inverse Plot", "Percentile must be a number between 0 and 100.")
            return
        if p < 0.0 or p > 100.0:
            messagebox.showerror("Inverse Plot", "Percentile must be between 0 and 100.")
            return
        self.inv_percentile_var.set(f"{p:g}")
        self._update_plot()

    def _current_inverse_percentile(self) -> float:
        text = self.inv_percentile_var.get().strip()
        try:
            p = float(text)
        except Exception:
            return 10.0
        return max(0.0, min(100.0, p))

    def _read_uncertainty_config(
        self,
        enabled_var: BooleanVar,
        t_var: StringVar,
        eps_var: StringVar,
        mu_var: StringVar,
    ) -> UncertaintyConfig:
        if not enabled_var.get():
            return UncertaintyConfig(enabled=False, thickness_pct=0.0, eps_pct=0.0, mu_pct=0.0)

        thickness_pct = float(t_var.get().strip())
        eps_pct = float(eps_var.get().strip())
        mu_pct = float(mu_var.get().strip())
        if not all(
            math.isfinite(value)
            for value in (thickness_pct, eps_pct, mu_pct)
        ):
            raise ValueError("Uncertainty percentages must be finite.")
        if thickness_pct < 0 or eps_pct < 0 or mu_pct < 0:
            raise ValueError("Uncertainty percentages must be >= 0.")
        if thickness_pct >= 100 or eps_pct >= 100 or mu_pct >= 100:
            raise ValueError(
                "Uncertainty percentages must be < 100 so all physical "
                "scales remain positive."
            )
        return UncertaintyConfig(
            enabled=True,
            thickness_pct=thickness_pct,
            eps_pct=eps_pct,
            mu_pct=mu_pct,
        )

    def _save_plot(self) -> None:
        if not MPL_AVAILABLE or self.fig is None:
            messagebox.showerror("Plot", "Matplotlib is not available.")
            return
        p = filedialog.asksaveasfilename(
            title="Save plot image",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg;*.jpeg"), ("All Files", "*.*")],
        )
        if not p:
            return
        try:
            self.fig.savefig(p, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Plot", f"Saved plot to:\n{p}")
        except Exception as exc:
            messagebox.showerror("Plot", str(exc))

    def _save_heatmap_only(self) -> None:
        if not MPL_AVAILABLE:
            messagebox.showerror("Heatmap", "Matplotlib is not available.")
            return
        view = self._active_heatmap_view() if self._is_heatmap_tab_active() else None
        if view is None:
            messagebox.showerror(
                "Heatmap",
                "No heatmap data to save. Run the Off Angle or Thickness compute first.",
            )
            return

        selected = self._get_selected_metric_grid(view)
        if selected is None:
            messagebox.showerror("Heatmap", "Select a valid heatmap metric first.")
            return
        metric_label, metric_key, z = selected

        try:
            cmin, cmax = self._get_color_limits()
        except Exception as exc:
            messagebox.showerror("Heatmap", str(exc))
            return

        p = filedialog.asksaveasfilename(
            title="Save heatmap image (no crosshairs)",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg;*.jpeg"), ("All Files", "*.*")],
        )
        if not p:
            return

        try:
            x_vals = view.x_vals
            freqs = view.freqs
            cmap = "magma" if "[span]" in metric_label else ("twilight" if "phase" in metric_key else "viridis")

            fig = Figure(figsize=(7.0, 4.8), dpi=100)
            ax = fig.add_subplot(111)
            im = ax.imshow(
                z,
                origin="lower",
                aspect="auto",
                extent=(x_vals[0], x_vals[-1], freqs[0], freqs[-1]),
                cmap=cmap,
                vmin=cmin,
                vmax=cmax,
            )
            ax.set_title(f"{metric_label} vs {view.x_name}/Frequency")
            ax.set_xlabel(view.x_label)
            ax.set_ylabel("Frequency (GHz)")
            ax.grid(False)
            self._style_plot_axis(ax)
            cbar = fig.colorbar(im, ax=ax)
            if cmin is None or cmax is None:
                cbar.set_label(metric_label)
            else:
                cbar.set_label(f"{metric_label} [{cmin:g}, {cmax:g}]")
            style_colorbar(cbar, self._colors)
            fig.patch.set_facecolor(self._colors["plot_bg"])
            fig.savefig(p, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Heatmap", f"Saved heatmap to:\n{p}")
        except Exception as exc:
            messagebox.showerror("Heatmap", str(exc))

    def _show_about(self) -> None:
        messagebox.showinfo(f"About {APP_ACRONYM}", ABOUT_TEXT)

    def _get_color_limits(self) -> tuple[float | None, float | None]:
        if self.cbar_auto_var.get():
            return None, None
        cmin_text = self.cbar_min_var.get().strip()
        cmax_text = self.cbar_max_var.get().strip()
        if not cmin_text or not cmax_text:
            raise ValueError("Set both colorbar Min and Max, or enable Auto color scale.")
        cmin = float(cmin_text)
        cmax = float(cmax_text)
        if cmax <= cmin:
            raise ValueError("Colorbar Max must be greater than Min.")
        return cmin, cmax

    def _stats(self, values: list[float]) -> tuple[float, float, float]:
        if not values:
            return float("nan"), float("nan"), float("nan")
        if NUMPY_AVAILABLE:
            arr = np.asarray(values, dtype=float)
            return float(arr.mean()), float(arr.min()), float(arr.max())
        mean = sum(values) / len(values)
        return mean, min(values), max(values)

    def _max_contiguous_bandwidth(
        self,
        freqs: list[float],
        values: list[float],
        threshold: float,
    ) -> float:
        if len(freqs) < 2:
            return 0.0
        best = 0.0
        run_start: int | None = None
        for i, v in enumerate(values):
            if v <= threshold:
                if run_start is None:
                    run_start = i
            elif run_start is not None:
                best = max(best, freqs[i - 1] - freqs[run_start])
                run_start = None
        if run_start is not None:
            best = max(best, freqs[-1] - freqs[run_start])
        return max(best, 0.0)

    def _summarize_angle_run(
        self,
        out: dict[str, list[list[float]] | list[float]],
        wave_pol: str,
        uncertainty_enabled: bool,
    ) -> str:
        freqs = out["freq_ghz"]
        angles = out["angle_deg"]
        metal = out["metal_loss_db"]
        air = out["air_loss_db"]
        ins = out["insertion_loss_db"]
        phase = out["metal_phase_deg"]
        metal_abs = out["metal_absorption_db"]

        all_metal = [v for row in metal for v in row]
        all_air = [v for row in air for v in row]
        all_ins = [v for row in ins for v in row]
        all_phase = [v for row in phase for v in row]
        all_metal_abs = [v for row in metal_abs for v in row]
        metal_mean, metal_min, metal_max = self._stats(all_metal)
        air_mean, air_min, air_max = self._stats(all_air)
        ins_mean, ins_min, ins_max = self._stats(all_ins)
        phase_mean, phase_min, phase_max = self._stats(all_phase)
        abs_mean, abs_min, abs_max = self._stats(all_metal_abs)

        band_threshold = -10.0
        best_bw = 0.0
        best_bw_angle = angles[0]
        for j, a in enumerate(angles):
            row = [metal[i][j] for i in range(len(freqs))]
            bw = self._max_contiguous_bandwidth(freqs, row, band_threshold)
            if bw > best_bw:
                best_bw = bw
                best_bw_angle = a

        best_angle = angles[0]
        best_angle_score = float("inf")
        for j, a in enumerate(angles):
            row = [metal[i][j] for i in range(len(freqs))]
            score, _mn, _mx = self._stats(row)
            if score < best_angle_score:
                best_angle_score = score
                best_angle = a

        unc_state = "ON" if uncertainty_enabled else "OFF"
        return (
            f"Mode: angle-frequency heatmap ({wave_pol.upper()}) | Uncertainty: {unc_state}\n"
            f"Grid: {len(freqs)} freq x {len(angles)} angle points\n"
            f"Metal loss dB mean/min/max: {metal_mean:.3f} / {metal_min:.3f} / {metal_max:.3f}\n"
            f"Metal absorption dB mean/min/max: {abs_mean:.3f} / {abs_min:.3f} / {abs_max:.3f}\n"
            f"Air loss dB mean/min/max: {air_mean:.3f} / {air_min:.3f} / {air_max:.3f}\n"
            f"Insertion loss dB mean/min/max: {ins_mean:.3f} / {ins_min:.3f} / {ins_max:.3f}\n"
            f"Metal phase deg mean/min/max: {phase_mean:.3f} / {phase_min:.3f} / {phase_max:.3f}\n"
            f"Best average metal loss angle: {best_angle:.2f} deg ({best_angle_score:.3f} dB)\n"
            f"Max contiguous bandwidth with metal loss <= {band_threshold:.0f} dB: "
            f"{best_bw:.3f} GHz @ {best_bw_angle:.2f} deg"
        )

    def _summarize_thickness_run(
        self,
        out: dict[str, list[list[float]] | list[float]],
        wave_pol: str,
        angle_deg: float,
        uncertainty_enabled: bool,
    ) -> str:
        freqs = out["freq_ghz"]
        thicknesses = out["thickness_in"]
        metal = out["metal_loss_db"]
        air = out["air_loss_db"]
        ins = out["insertion_loss_db"]
        metal_abs = out["metal_absorption_db"]

        metal_mean, metal_min, metal_max = self._stats([v for row in metal for v in row])
        air_mean, air_min, air_max = self._stats([v for row in air for v in row])
        ins_mean, ins_min, ins_max = self._stats([v for row in ins for v in row])
        abs_mean, abs_min, abs_max = self._stats([v for row in metal_abs for v in row])

        # Per-thickness figures of merit: average metal loss across the band and
        # the widest contiguous band under -10 dB.
        band_threshold = -10.0
        best_t = thicknesses[0]
        best_t_score = float("inf")
        best_bw = 0.0
        best_bw_t = thicknesses[0]
        for j, t_in in enumerate(thicknesses):
            column = [metal[i][j] for i in range(len(freqs))]
            score, _mn, _mx = self._stats(column)
            if score < best_t_score:
                best_t_score = score
                best_t = t_in
            bw = self._max_contiguous_bandwidth(freqs, column, band_threshold)
            if bw > best_bw:
                best_bw = bw
                best_bw_t = t_in

        unc_state = "ON" if uncertainty_enabled else "OFF"
        return (
            f"Mode: thickness-frequency heatmap ({wave_pol.upper()} @ {angle_deg:g} deg) | "
            f"Uncertainty: {unc_state}\n"
            f"Grid: {len(freqs)} freq x {len(thicknesses)} thickness points "
            f"({thicknesses[0]:g} to {thicknesses[-1]:g} in)\n"
            f"Metal loss dB mean/min/max: {metal_mean:.3f} / {metal_min:.3f} / {metal_max:.3f}\n"
            f"Metal absorption dB mean/min/max: {abs_mean:.3f} / {abs_min:.3f} / {abs_max:.3f}\n"
            f"Air loss dB mean/min/max: {air_mean:.3f} / {air_min:.3f} / {air_max:.3f}\n"
            f"Insertion loss dB mean/min/max: {ins_mean:.3f} / {ins_min:.3f} / {ins_max:.3f}\n"
            f"Best average metal loss thickness: {best_t:g} in ({best_t_score:.3f} dB)\n"
            f"Max contiguous bandwidth with metal loss <= {band_threshold:.0f} dB: "
            f"{best_bw:.3f} GHz @ {best_bw_t:g} in"
        )

    def _summarize_frequency_run(
        self,
        sweep: list[float],
        loaded_layers: list[LoadedLayer],
        wave_pol: str,
        uncertainty_enabled: bool,
        backing: str,
    ) -> str:
        metrics = compute_angle_metrics_many(sweep, 0.0, loaded_layers, wave_pol)
        metal = metrics["metal_loss_db"]
        air = metrics["air_loss_db"]
        ins = metrics["insertion_loss_db"]
        phase = metrics["metal_phase_deg"]
        metal_abs = metrics["metal_absorption_db"]
        metal_mean, metal_min, metal_max = self._stats(metal)
        air_mean, air_min, air_max = self._stats(air)
        ins_mean, ins_min, ins_max = self._stats(ins)
        phase_mean, phase_min, phase_max = self._stats(phase)
        abs_mean, abs_min, abs_max = self._stats(metal_abs)
        bw10 = self._max_contiguous_bandwidth(sweep, metal, -10.0)
        unc_state = "ON" if uncertainty_enabled else "OFF"
        return (
            f"Mode: frequency sweep ({wave_pol.upper()}, backing={backing}) | Uncertainty: {unc_state}\n"
            f"Points: {len(sweep)}\n"
            f"Metal loss dB mean/min/max: {metal_mean:.3f} / {metal_min:.3f} / {metal_max:.3f}\n"
            f"Metal absorption dB mean/min/max: {abs_mean:.3f} / {abs_min:.3f} / {abs_max:.3f}\n"
            f"Air loss dB mean/min/max: {air_mean:.3f} / {air_min:.3f} / {air_max:.3f}\n"
            f"Insertion loss dB mean/min/max: {ins_mean:.3f} / {ins_min:.3f} / {ins_max:.3f}\n"
            f"Metal phase deg mean/min/max: {phase_mean:.3f} / {phase_min:.3f} / {phase_max:.3f}\n"
            f"Contiguous bandwidth with metal loss <= -10 dB at 0 deg: {bw10:.3f} GHz"
        )

    def _selected_idx(self) -> int | None:
        row = self.layer_list.currentRow()
        if row < 0:
            return None
        return int(row)

    def _draw_layer_preview(self, painter: QPainter) -> None:
        colors = self._colors
        width = self.layer_preview.width()
        height = self.layer_preview.height()

        # The LayerPreview widget paints its own surface, so start by filling it.
        painter.fillRect(0, 0, width, height, QColor(colors["preview_bg"]))
        if width < 40 or height < 40:
            return

        base_font = painter.font()
        line_h = painter.fontMetrics().height()

        if not self.layers:
            painter.setPen(QColor(colors["preview_empty"]))
            painter.drawText(QRectF(0, 0, width, height), Qt.AlignCenter, "No layers configured")
            return

        pad = 12.0
        title_gap = 18.0
        x0 = pad
        x1 = width - pad
        y0 = pad + title_gap
        y1 = height - pad - title_gap
        if y1 <= y0:
            return

        painter.setPen(QColor(colors["preview_text"]))
        painter.drawText(
            QRectF(x0, pad, x1 - x0, title_gap),
            Qt.AlignHCenter | Qt.AlignTop,
            "Top (incident side)",
        )
        painter.drawText(
            QRectF(x0, height - pad - title_gap, x1 - x0, title_gap),
            Qt.AlignHCenter | Qt.AlignBottom,
            "Bottom / backing",
        )

        thicknesses = [max(layer.thickness_in, 0.0) if not layer.is_sheet else 0.0 for layer in self.layers]
        n = len(self.layers)
        n_bulk = sum(1 for layer in self.layers if not layer.is_sheet)
        n_sheet = n - n_bulk
        sheet_h = 6.0
        stack_h = y1 - y0 - n_sheet * sheet_h
        if n_bulk > 0:
            min_h = min(22.0, stack_h / max(float(n_bulk), 1.0))
        else:
            min_h = 0.0
        min_total = min_h * n_bulk

        if n_bulk == 0 or stack_h <= min_total or sum(thicknesses) <= 0:
            bulk_h_each = stack_h / max(n_bulk, 1)
            heights = [sheet_h if layer.is_sheet else bulk_h_each for layer in self.layers]
        else:
            extra_h = stack_h - min_total
            total_t = sum(thicknesses) or 1.0
            heights = [
                sheet_h if layer.is_sheet else min_h + extra_h * (t / total_t)
                for layer, t in zip(self.layers, thicknesses)
            ]

        layer_colors = colors["layer_colors"]

        y = y0
        for i, (layer, layer_h) in enumerate(zip(self.layers, heights), start=1):
            yn = y1 if i == n else y + layer_h
            mid = (y + yn) * 0.5

            if layer.is_sheet:
                pen = QPen(QColor(colors.get("accent", "#3b82f6")))
                pen.setWidth(2)
                pen.setStyle(Qt.CustomDashLine)
                pen.setDashPattern([6, 3])
                painter.setPen(pen)
                painter.drawLine(QPointF(x0, mid), QPointF(x1, mid))
                label = f"{i}. SHEET {layer.sheet_resistance:g} \u03a9/sq"
                max_chars = max(16, int((x1 - x0) / 6.7))
                if len(label) > max_chars:
                    label = label[: max_chars - 3] + "..."
                small_font = QFont(base_font)
                small_font.setPointSize(8)
                painter.setFont(small_font)
                small_h = painter.fontMetrics().height()
                painter.setPen(QColor(colors["preview_layer_text"]))
                painter.drawText(
                    QRectF(x0, mid - 7 - small_h / 2.0, x1 - x0, small_h),
                    Qt.AlignHCenter | Qt.AlignVCenter,
                    label,
                )
                painter.setFont(base_font)
                y = yn
                continue

            fill = layer_colors[(i - 1) % len(layer_colors)]
            rect = QRectF(x0, y, x1 - x0, yn - y)
            painter.fillRect(rect, QColor(fill))
            pen = QPen(QColor(colors["preview_layer_border"]))
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawRect(rect)

            material_name = Path(layer.file_0deg).stem or Path(layer.file_0deg).name or "material"
            layer_type = "aniso" if layer.anisotropic else "iso"
            label = f"{i}. {material_name} | {layer.thickness_in:g} in | {layer_type}"
            max_chars = max(16, int((x1 - x0) / 6.7))
            if len(label) > max_chars:
                label = label[: max_chars - 3] + "..."
            painter.setPen(QColor(colors["preview_layer_text"]))
            painter.drawText(
                QRectF(x0, mid - line_h / 2.0, x1 - x0, line_h),
                Qt.AlignHCenter | Qt.AlignVCenter,
                label,
            )
            y = yn

        pen = QPen(QColor(colors["preview_outline"]))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawRect(QRectF(x0, y0, x1 - x0, y1 - y0))

    def _refresh_layers(self) -> None:
        self.layer_list.clear()
        for i, layer in enumerate(self.layers, start=1):
            if layer.is_sheet:
                desc = f"{i}. SHEET R={layer.sheet_resistance:g} \u03a9/sq"
                if layer.inv_rs_min is not None or layer.inv_rs_max is not None:
                    rparts = []
                    if layer.inv_rs_min is not None:
                        rparts.append(f"min={layer.inv_rs_min:g}")
                    if layer.inv_rs_max is not None:
                        rparts.append(f"max={layer.inv_rs_max:g}")
                    if layer.inv_rs_accuracy is not None:
                        rparts.append(f"acc={layer.inv_rs_accuracy:g}")
                    desc += f" | inv[{', '.join(rparts)}]"
            elif layer.anisotropic:
                file0 = Path(layer.file_0deg).name or layer.file_0deg
                file90 = Path(layer.file_90deg).name or layer.file_90deg
                desc = (
                    f"{i}. t={layer.thickness_in:g} in | aniso | pol={layer.polarization_deg:g} deg | "
                    f"0deg={file0} | 90deg={file90}"
                )
            else:
                file0 = Path(layer.file_0deg).name or layer.file_0deg
                desc = f"{i}. t={layer.thickness_in:g} in | iso | file={file0}"
            if not layer.is_sheet and (
                layer.inv_t_min_in is not None
                or layer.inv_t_max_in is not None
                or layer.inv_t_accuracy_in is not None
            ):
                parts = []
                if layer.inv_t_min_in is not None:
                    parts.append(f"min={layer.inv_t_min_in:g}")
                if layer.inv_t_max_in is not None:
                    parts.append(f"max={layer.inv_t_max_in:g}")
                if layer.inv_t_accuracy_in is not None:
                    parts.append(f"acc={layer.inv_t_accuracy_in:g}")
                desc += f" | inv[{', '.join(parts)}]"
            self.layer_list.addItem(desc)
        self.layer_preview.update()
        self._refresh_thickness_layers()

    def _thickness_layer_choices(self) -> list[tuple[int, str]]:
        """Layers whose thickness can be swept, as (index, label) pairs. Sheet
        layers are excluded: they are zero-thickness impedance boundaries."""
        return [
            (i, f"{i + 1}. {Path(layer.file_0deg).name or layer.file_0deg}")
            for i, layer in enumerate(self.layers)
            if not layer.is_sheet
        ]

    def _refresh_thickness_layers(self) -> None:
        labels = [label for _idx, label in self._thickness_layer_choices()]
        for combo, var in (
            (self.thk_layer_combo, self.thk_layer_var),
            (self.ibc_batch_layer_combo, self.ibc_batch_layer_var),
        ):
            if combo is None:
                continue
            previous = var.get()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(labels)
            selected = previous if previous in labels else (labels[0] if labels else "")
            if selected:
                combo.setCurrentIndex(labels.index(selected))
            combo.blockSignals(False)
            var.set(selected)
        self._refresh_ibc_batch_preview()

    def _selected_thickness_layer_index(self) -> int:
        choices = self._thickness_layer_choices()
        if not choices:
            raise ValueError(
                "Add at least one material layer (sheet layers have no thickness to sweep)."
            )
        wanted = self.thk_layer_var.get().strip()
        for idx, label in choices:
            if label == wanted:
                return idx
        return choices[0][0]

    def _selected_ibc_batch_layer_index(self) -> int:
        choices = self._thickness_layer_choices()
        if not choices:
            raise ValueError(
                "Add at least one material layer; sheet layers have no thickness."
            )
        wanted = self.ibc_batch_layer_var.get().strip()
        for idx, label in choices:
            if label == wanted:
                return idx
        raise ValueError("Select a valid material layer for the IBC batch.")

    def _plan_ibc_batch(self) -> list[IbcBatchItem]:
        return plan_ibc_thickness_batch(
            self.ibc_batch_output_dir_var.get(),
            self.ibc_batch_prefix_var.get(),
            self.ibc_batch_start_var.get(),
            self.ibc_batch_stop_var.get(),
            self.ibc_batch_step_var.get(),
            self.ibc_batch_unit_var.get(),
        )

    def _refresh_ibc_batch_preview(self) -> None:
        if self.ibc_batch_preview_label is None:
            return
        try:
            self._selected_ibc_batch_layer_index()
            plan = self._plan_ibc_batch()
            first = plan[0].path.name
            last = plan[-1].path.name
            filenames = first if len(plan) == 1 else f"{first} … {last}"
            self.ibc_batch_preview_label.setText(
                f"Preflight: {len(plan)} nominal PEC-backed IBC file(s) — {filenames}"
            )
            if self.ibc_batch_export_btn is not None:
                self.ibc_batch_export_btn.setText(
                    f"Export {len(plan)} IBC file(s)"
                )
                self.ibc_batch_export_btn.setEnabled(not self._task_running)
        except Exception as exc:
            self.ibc_batch_preview_label.setText(f"Preflight: {exc}")
            if self.ibc_batch_export_btn is not None:
                self.ibc_batch_export_btn.setText("Export IBC batch")
                self.ibc_batch_export_btn.setEnabled(False)

    def _add_layer(self) -> None:
        dlg = LayerDialog(self, presets=BUILTIN_MATERIAL_PRESETS)
        dlg.exec()
        if dlg.result is not None:
            self.layers.append(dlg.result)
            self._refresh_layers()

    def _add_sheet(self) -> None:
        dlg = SheetDialog(self)
        dlg.exec()
        if dlg.result is not None:
            self.layers.append(dlg.result)
            self._refresh_layers()

    def _edit_layer(self) -> None:
        idx = self._selected_idx()
        if idx is None:
            messagebox.showwarning("Layer", "Select a layer to edit.")
            return
        layer = self.layers[idx]
        if layer.is_sheet:
            dlg = SheetDialog(self, initial=layer)
            dlg.exec()
            if dlg.result is not None:
                self.layers[idx] = dlg.result
                self._refresh_layers()
                self.layer_list.setCurrentRow(idx)
        else:
            dlg = LayerDialog(self, layer, presets=BUILTIN_MATERIAL_PRESETS)
            dlg.exec()
            if dlg.result is not None:
                self.layers[idx] = dlg.result
                self._refresh_layers()
                self.layer_list.setCurrentRow(idx)

    def _remove_layer(self) -> None:
        idx = self._selected_idx()
        if idx is None:
            messagebox.showwarning("Layer", "Select a layer to remove.")
            return
        del self.layers[idx]
        self._refresh_layers()

    def _move_up(self) -> None:
        idx = self._selected_idx()
        if idx is None or idx == 0:
            return
        self.layers[idx - 1], self.layers[idx] = self.layers[idx], self.layers[idx - 1]
        self._refresh_layers()
        self.layer_list.setCurrentRow(idx - 1)

    def _move_down(self) -> None:
        idx = self._selected_idx()
        if idx is None or idx >= len(self.layers) - 1:
            return
        self.layers[idx + 1], self.layers[idx] = self.layers[idx], self.layers[idx + 1]
        self._refresh_layers()
        self.layer_list.setCurrentRow(idx + 1)

    def _load_layers(self, skiprows: int, layer_configs: list[LayerConfig] | None = None) -> list[LoadedLayer]:
        source_layers = self.layers if layer_configs is None else layer_configs
        loaded: list[LoadedLayer] = []
        for i, layer in enumerate(source_layers, start=1):
            if layer.is_sheet:
                if layer.sheet_resistance <= 0:
                    raise ValueError(f"Layer {i}: sheet resistance must be > 0.")
                loaded.append(
                    LoadedLayer(
                        thickness_m=0.0,
                        anisotropic=False,
                        polarization_deg=0.0,
                        table_0deg=None,
                        table_90deg=None,
                        is_sheet=True,
                        sheet_resistance=layer.sheet_resistance,
                    )
                )
                continue

            t_m = layer.thickness_in * INCH_TO_M
            if t_m <= 0:
                raise ValueError(f"Layer {i}: thickness must be > 0.")

            table_0 = read_material_table(Path(layer.file_0deg), skiprows)
            table_90 = (
                read_material_table(Path(layer.file_90deg), skiprows)
                if layer.anisotropic
                else None
            )
            loaded.append(
                LoadedLayer(
                    thickness_m=t_m,
                    anisotropic=layer.anisotropic,
                    polarization_deg=layer.polarization_deg,
                    table_0deg=table_0,
                    table_90deg=table_90,
                )
            )
        return loaded

    def _snapshot_layers(self) -> list[LayerConfig]:
        return [
            LayerConfig(
                thickness_in=layer.thickness_in,
                anisotropic=layer.anisotropic,
                file_0deg=layer.file_0deg,
                file_90deg=layer.file_90deg,
                polarization_deg=layer.polarization_deg,
                is_sheet=layer.is_sheet,
                sheet_resistance=layer.sheet_resistance,
                inv_t_min_in=layer.inv_t_min_in,
                inv_t_max_in=layer.inv_t_max_in,
                inv_t_accuracy_in=layer.inv_t_accuracy_in,
                inv_rs_min=layer.inv_rs_min,
                inv_rs_max=layer.inv_rs_max,
                inv_rs_accuracy=layer.inv_rs_accuracy,
            )
            for layer in self.layers
        ]

    def _set_task_state(self, running: bool, text: str) -> None:
        self._task_running = running
        for btn in (
            self.compute_btn,
            self.ibc_batch_export_btn,
            self.angle_compute_btn,
            self.thk_compute_btn,
            self.inv_run_btn,
            self.inv_apply_btn,
            self.layer_add_btn,
            self.layer_add_sheet_btn,
            self.layer_edit_btn,
            self.layer_remove_btn,
            self.layer_up_btn,
            self.layer_down_btn,
            self.mix_add_btn,
            self.mix_edit_btn,
            self.mix_remove_btn,
            self.mix_run_btn,
            self.mix_preview_btn,
            self.mix_apply_btn,
            self.mix_export_btn,
        ):
            if btn is not None:
                btn.setEnabled(not running)
        self._refresh_ibc_batch_preview()
        self.status_var.set(text)
        if self.status_progress is not None:
            self.status_progress.setVisible(running)

    def _run_background_task(
        self,
        task_name: str,
        worker: Callable[[], _T],
        on_success: Callable[[_T], None],
        error_title: str,
    ) -> None:
        if self._task_running:
            messagebox.showwarning(task_name, "Another task is already running.")
            return

        result_q: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)
        self._set_task_state(True, f"{task_name} running...")

        def _runner() -> None:
            try:
                result_q.put(("ok", worker()))
            except Exception as exc:
                result_q.put(("err", exc))

        threading.Thread(target=_runner, daemon=True).start()

        timer = QTimer(self)
        timer.setInterval(100)

        def _poll() -> None:
            try:
                status, payload = result_q.get_nowait()
            except queue.Empty:
                return
            timer.stop()
            self._set_task_state(False, "Ready")
            if status == "ok":
                on_success(payload)  # type: ignore[arg-type]
            else:
                messagebox.showerror(error_title, str(payload))

        timer.timeout.connect(_poll)
        timer.start()

    def _select_mode(self, index: int) -> None:
        if self.mode_stack is not None:
            self.mode_stack.setCurrentIndex(index)
        self._on_left_tab_changed(None)

    def _active_left_tab_label(self) -> str:
        if self.mode_stack is None:
            return ""
        try:
            return self._mode_labels[self.mode_stack.currentIndex()]
        except Exception:
            return ""

    def _is_angle_tab_active(self) -> bool:
        return self._active_left_tab_label() == "Off Angle"

    def _is_inverse_tab_active(self) -> bool:
        return self._active_left_tab_label() == "Inverse Design"

    def _is_mix_tab_active(self) -> bool:
        return self._active_left_tab_label() == "Material Mix"

    def _is_thickness_tab_active(self) -> bool:
        return self._active_left_tab_label() == "Thickness"

    def _is_heatmap_tab_active(self) -> bool:
        return self._is_angle_tab_active() or self._is_thickness_tab_active()

    def _active_heatmap_view(self) -> HeatmapView | None:
        """The computed heatmap belonging to the active tab, if any. Off Angle
        and Thickness results are kept separately so switching tabs does not
        discard the other view's run."""
        if self._is_thickness_tab_active():
            if self.last_thickness_results is None:
                return None
            return HeatmapView(
                results=self.last_thickness_results,
                unc_min=self.last_thickness_uncertainty_min,
                unc_max=self.last_thickness_uncertainty_max,
                x_key="thickness_in",
                x_name="Thickness",
                x_unit="in",
            )
        if self.last_heatmap_results is None:
            return None
        return HeatmapView(
            results=self.last_heatmap_results,
            unc_min=self.last_heatmap_uncertainty_min,
            unc_max=self.last_heatmap_uncertainty_max,
            x_key="angle_deg",
            x_name="Angle",
            x_unit="deg",
        )

    def _on_left_tab_changed(self, _event: object) -> None:
        # Slice indices address whichever grid is on screen; drop them so a
        # stale angle index is not reused as a thickness index.
        self.selected_x_idx = None
        self.selected_freq_idx = None
        if self.slice_x_label is not None:
            view_name = "Thickness slice (in)" if self._is_thickness_tab_active() else "Angle slice (deg)"
            self.slice_x_label.setText(view_name)
        self._update_plot()

    def _get_selected_metric_grid(
        self, view: HeatmapView
    ) -> tuple[str, str, list[list[float]]] | None:
        metric_label = self.heatmap_metric_var.get()
        metric_key = self.metric_label_to_key.get(metric_label)
        if metric_key is None:
            return None

        view_key = self.uncertainty_view_label_to_key.get(
            self.uncertainty_view_var.get(),
            "nominal",
        )
        base = view.results[metric_key]
        if view_key == "nominal":
            return metric_label, metric_key, base

        if view.unc_min is None or view.unc_max is None:
            return metric_label, metric_key, base

        z_min = view.unc_min[metric_key]
        z_max = view.unc_max[metric_key]
        if view_key == "min":
            return f"{metric_label} [min]", metric_key, z_min
        if view_key == "max":
            return f"{metric_label} [max]", metric_key, z_max
        if NUMPY_AVAILABLE:
            span = (np.asarray(z_max, dtype=float) - np.asarray(z_min, dtype=float)).tolist()
        else:
            span = [
                [z_max[i][j] - z_min[i][j] for j in range(len(z_min[i]))]
                for i in range(len(z_min))
            ]
        return f"{metric_label} [span]", metric_key, span

    def _apply_manual_slices(self) -> None:
        if not self._is_heatmap_tab_active():
            messagebox.showwarning(
                "Slice Input", "Switch to the Off Angle or Thickness view to edit heatmap slices."
            )
            return
        view = self._active_heatmap_view()
        if view is None:
            messagebox.showwarning(
                "Slice Input", f"Run the {self._active_left_tab_label()} compute first."
            )
            return

        x_vals = view.x_vals
        freqs = view.freqs
        angle_text = self.slice_angle_var.get().strip()
        freq_text = self.slice_freq_var.get().strip()
        if not angle_text and not freq_text:
            return

        try:
            if angle_text:
                x_val = float(angle_text)
                if x_val < x_vals[0] or x_val > x_vals[-1]:
                    raise ValueError(
                        f"{view.x_name} slice must be in "
                        f"[{x_vals[0]:g}, {x_vals[-1]:g}] {view.x_unit}."
                    )
                self.selected_x_idx = nearest_index(x_vals, x_val)
            if freq_text:
                freq_val = float(freq_text)
                if freq_val < freqs[0] or freq_val > freqs[-1]:
                    raise ValueError(f"Frequency slice must be in [{freqs[0]:g}, {freqs[-1]:g}] GHz.")
                self.selected_freq_idx = nearest_index(freqs, freq_val)
        except Exception as exc:
            messagebox.showerror("Slice Input", str(exc))
            return

        self._update_plot()

    def _update_slice_plots(
        self,
        view: HeatmapView,
        z: list[list[float]],
        metric_label: str,
    ) -> None:
        colors = self._colors
        x_vals = view.x_vals
        freqs = view.freqs
        if (
            self.ax_freq_slice is None
            or self.ax_angle_slice is None
            or self.ax_heatmap is None
            or not x_vals
            or not freqs
        ):
            return

        if self.selected_x_idx is None or self.selected_x_idx >= len(x_vals):
            self.selected_x_idx = len(x_vals) // 2
        if self.selected_freq_idx is None or self.selected_freq_idx >= len(freqs):
            self.selected_freq_idx = len(freqs) // 2

        j = self.selected_x_idx
        i = self.selected_freq_idx
        x_sel = x_vals[j]
        freq_sel = freqs[i]
        self.slice_angle_var.set(f"{x_sel:.6g}")
        self.slice_freq_var.set(f"{freq_sel:.6g}")

        freq_slice = [row[j] for row in z]
        x_slice = z[i]

        self.ax_freq_slice.clear()
        self.ax_freq_slice.plot(freqs, freq_slice, color=colors["plot_line_freq"], linewidth=1.7)
        self.ax_freq_slice.set_title(
            f"Frequency Slice @ {x_sel:g} {view.x_unit}",
            fontsize=9,
            pad=2,
        )
        self.ax_freq_slice.set_xlabel("Frequency (GHz)", fontsize=8)
        self.ax_freq_slice.set_ylabel(metric_label, fontsize=8)
        self._style_plot_axis(self.ax_freq_slice)
        self.ax_freq_slice.grid(True, color=colors["plot_grid"], alpha=0.3)

        self.ax_angle_slice.clear()
        self.ax_angle_slice.plot(x_vals, x_slice, color=colors["plot_line_angle"], linewidth=1.7)
        self.ax_angle_slice.set_title(
            f"{view.x_name} Slice @ {freq_sel:g} GHz",
            fontsize=9,
            pad=2,
        )
        self.ax_angle_slice.set_xlabel(view.x_label, fontsize=8)
        self._style_plot_axis(self.ax_angle_slice)
        self.ax_angle_slice.grid(True, color=colors["plot_grid"], alpha=0.3)

        if view.unc_min is not None and view.unc_max is not None:
            metric_key = self.metric_label_to_key[self.heatmap_metric_var.get()]
            zmin = view.unc_min[metric_key]
            zmax = view.unc_max[metric_key]
            self.ax_freq_slice.fill_between(
                freqs,
                [r[j] for r in zmin],
                [r[j] for r in zmax],
                color=colors["plot_line_freq"],
                alpha=0.12,
                linewidth=0,
            )
            self.ax_angle_slice.fill_between(
                x_vals,
                zmin[i],
                zmax[i],
                color=colors["plot_line_angle"],
                alpha=0.12,
                linewidth=0,
            )

        self.ax_heatmap.axvline(
            x_sel,
            color=colors["plot_crosshair"],
            linewidth=1.0,
            linestyle="--",
            alpha=0.9,
        )
        self.ax_heatmap.axhline(
            freq_sel,
            color=colors["plot_crosshair"],
            linewidth=1.0,
            linestyle="--",
            alpha=0.9,
        )

    def _on_plot_click(self, event: object) -> None:
        if (
            not MPL_AVAILABLE
            or self.ax_heatmap is None
            or not self._is_heatmap_tab_active()
            or getattr(event, "inaxes", None) is not self.ax_heatmap
            or getattr(event, "xdata", None) is None
            or getattr(event, "ydata", None) is None
        ):
            return
        view = self._active_heatmap_view()
        if view is None:
            return

        x = float(event.xdata)
        y = float(event.ydata)
        self.selected_x_idx = nearest_index(view.x_vals, x)
        self.selected_freq_idx = nearest_index(view.freqs, y)
        self._update_plot()

    def _draw_inverse_placeholder(self, text: str) -> None:
        if not MPL_AVAILABLE or self.ax_heatmap is None or self.canvas is None:
            return
        if self.heatmap_cbar is not None:
            self.heatmap_cbar.remove()
            self.heatmap_cbar = None
        colors = self._colors
        self.ax_heatmap.clear()
        self.ax_heatmap.set_title("Inverse Candidate Analysis")
        self.ax_heatmap.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            transform=self.ax_heatmap.transAxes,
            color=colors["muted_text"],
        )
        self._style_plot_axis(self.ax_heatmap)
        self.ax_heatmap.grid(False)
        if self.ax_freq_slice is not None:
            self.ax_freq_slice.clear()
            self.ax_freq_slice.set_title("Score vs Total Thickness", fontsize=9, pad=2)
            self.ax_freq_slice.set_xlabel("Total thickness (in)", fontsize=8)
            self.ax_freq_slice.set_ylabel("Score (dB)", fontsize=8)
            self._style_plot_axis(self.ax_freq_slice)
            self.ax_freq_slice.grid(False)
        if self.ax_angle_slice is not None:
            self.ax_angle_slice.clear()
            self.ax_angle_slice.set_title("Robustness Gap", fontsize=9, pad=2)
            self.ax_angle_slice.set_xlabel("Candidate rank", fontsize=8)
            self.ax_angle_slice.set_ylabel("Worst - Nominal (dB)", fontsize=8)
            self._style_plot_axis(self.ax_angle_slice)
            self.ax_angle_slice.grid(False)
        self.canvas.draw_idle()

    def _update_inverse_plot(self) -> None:
        if (
            not MPL_AVAILABLE
            or self.ax_heatmap is None
            or self.ax_freq_slice is None
            or self.ax_angle_slice is None
            or self.canvas is None
        ):
            return
        if self.heatmap_cbar is not None:
            self.heatmap_cbar.remove()
            self.heatmap_cbar = None
        if not self.inverse_candidates:
            self._draw_inverse_placeholder("Run inverse design to compare candidate stackups.")
            return

        colors = self._colors
        n = len(self.inverse_candidates)
        if (
            len(self.inverse_plot_samples) != n
            or not self.inverse_plot_freqs
        ):
            self._draw_inverse_placeholder("Run inverse design to compute percentile-vs-frequency curves.")
            return

        ranks = list(range(1, n + 1))
        scores = [c.score_db for c in self.inverse_candidates]
        worst = [c.worst_mean_db for c in self.inverse_candidates]
        nominal = [c.nominal_mean_db for c in self.inverse_candidates]
        total_thickness = [sum(c.thickness_in) for c in self.inverse_candidates]
        robustness_gap = [c.worst_mean_db - c.nominal_mean_db for c in self.inverse_candidates]

        selected_idx = 0
        if self.inv_results_list is not None:
            row = self.inv_results_list.currentRow()
            if row >= 0:
                selected_idx = int(row)
        selected_idx = max(0, min(selected_idx, n - 1))

        percentile = self._current_inverse_percentile()

        def _percentile(vals: list[float], p: float) -> float:
            if not vals:
                return float("nan")
            if NUMPY_AVAILABLE:
                return float(np.percentile(np.asarray(vals, dtype=float), p))
            sorted_vals = sorted(vals)
            if len(sorted_vals) == 1:
                return float(sorted_vals[0])
            pos = (p / 100.0) * (len(sorted_vals) - 1)
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            if lo == hi:
                return float(sorted_vals[lo])
            t = pos - lo
            return float(sorted_vals[lo] * (1.0 - t) + sorted_vals[hi] * t)

        curves: list[list[float]] = []
        for cand_samples in self.inverse_plot_samples:
            curve = [_percentile(freq_vals, percentile) for freq_vals in cand_samples]
            curves.append(curve)

        self.ax_heatmap.clear()
        freqs = self.inverse_plot_freqs
        for i, curve in enumerate(curves):
            if i == selected_idx:
                continue
            self.ax_heatmap.plot(freqs, curve, color=colors["plot_line_freq"], linewidth=1.0, alpha=0.25)
        self.ax_heatmap.plot(
            freqs,
            curves[selected_idx],
            color=colors["plot_line_angle"],
            linewidth=2.2,
            marker="o",
            markersize=3,
            label=f"Selected candidate (P{percentile:g})",
        )
        self.ax_heatmap.set_title(f"Metal Loss vs Frequency at P{percentile:g} across analyzed points")
        self.ax_heatmap.set_xlabel("Frequency (GHz)")
        self.ax_heatmap.set_ylabel("Metal loss (dB)")
        self._style_plot_axis(self.ax_heatmap)
        self.ax_heatmap.grid(True, color=colors["plot_grid"], alpha=0.3)
        self.ax_heatmap.legend(loc="best", fontsize=8)

        self.ax_freq_slice.clear()
        self.ax_freq_slice.scatter(total_thickness, scores, color=colors["plot_line_freq"], s=24, alpha=0.9)
        self.ax_freq_slice.scatter(
            [total_thickness[selected_idx]],
            [scores[selected_idx]],
            color=colors["plot_line_angle"],
            s=54,
            marker="*",
            zorder=3,
        )
        self.ax_freq_slice.set_title("Score vs Total Thickness", fontsize=9, pad=2)
        self.ax_freq_slice.set_xlabel("Total thickness (in)", fontsize=8)
        self.ax_freq_slice.set_ylabel("Score (dB)", fontsize=8)
        self._style_plot_axis(self.ax_freq_slice)
        self.ax_freq_slice.grid(True, color=colors["plot_grid"], alpha=0.3)

        self.ax_angle_slice.clear()
        self.ax_angle_slice.bar(ranks, robustness_gap, color=colors["plot_line_angle"], alpha=0.75)
        self.ax_angle_slice.bar(
            [ranks[selected_idx]],
            [robustness_gap[selected_idx]],
            color=colors["accent"],
            alpha=0.95,
        )
        self.ax_angle_slice.set_title("Robustness Gap (Worst - Nominal)", fontsize=9, pad=2)
        self.ax_angle_slice.set_xlabel("Candidate rank", fontsize=8)
        self.ax_angle_slice.set_ylabel("Gap (dB)", fontsize=8)
        self._style_plot_axis(self.ax_angle_slice)
        self.ax_angle_slice.grid(True, color=colors["plot_grid"], alpha=0.3)

        cand = self.inverse_candidates[selected_idx]
        mat_names = ", ".join(Path(p).name for p in cand.material_files)
        if len(mat_names) > 48:
            mat_names = mat_names[:45] + "..."
        self.ax_angle_slice.text(
            0.02,
            0.98,
            f"#{selected_idx + 1} score={cand.score_db:.3f} dB\n"
            f"nom={cand.nominal_mean_db:.3f}, worst={cand.worst_mean_db:.3f}\n"
            f"materials: {mat_names}",
            transform=self.ax_angle_slice.transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            color=colors["text"],
        )
        self.canvas.draw_idle()

    def _draw_plot_placeholder(self, text: str) -> None:
        if not MPL_AVAILABLE or self.ax_heatmap is None or self.canvas is None:
            return
        colors = self._colors
        x_label = "Thickness (in)" if self._is_thickness_tab_active() else "Angle (deg)"
        if self.heatmap_cbar is not None:
            self.heatmap_cbar.remove()
            self.heatmap_cbar = None
        self.ax_heatmap.clear()
        self.ax_heatmap.set_title("Heatmap")
        self.ax_heatmap.set_xlabel(x_label)
        self.ax_heatmap.set_ylabel("Frequency (GHz)")
        self.ax_heatmap.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            transform=self.ax_heatmap.transAxes,
            color=colors["muted_text"],
        )
        self._style_plot_axis(self.ax_heatmap)
        self.ax_heatmap.grid(False)
        if self.ax_freq_slice is not None:
            self.ax_freq_slice.clear()
            self.ax_freq_slice.set_title("Metric vs Frequency", fontsize=9, pad=2)
            self.ax_freq_slice.set_xlabel("Frequency (GHz)", fontsize=8)
            self.ax_freq_slice.set_ylabel("Metric", fontsize=8)
            self._style_plot_axis(self.ax_freq_slice)
            self.ax_freq_slice.grid(False)
        if self.ax_angle_slice is not None:
            self.ax_angle_slice.clear()
            self.ax_angle_slice.set_title(f"Metric vs {x_label.split(' (')[0]}", fontsize=9, pad=2)
            self.ax_angle_slice.set_xlabel(x_label, fontsize=8)
            self._style_plot_axis(self.ax_angle_slice)
            self.ax_angle_slice.grid(False)
        self.canvas.draw_idle()

    def _update_plot(self) -> None:
        if not MPL_AVAILABLE or self.ax_heatmap is None or self.canvas is None:
            return
        if self.plot_frame is not None:
            if self._is_inverse_tab_active():
                self.plot_frame.setTitle("Inverse Candidate Plots")
                self._update_inverse_plot()
                return
            if self._is_mix_tab_active():
                self.plot_frame.setTitle("Material Mix")
                self._update_mix_plot()
                return
            self.plot_frame.setTitle("Heatmap")
            if not self._is_heatmap_tab_active():
                self._draw_plot_placeholder(
                    "Heatmap and slice plots are shown in the Off Angle and Thickness views."
                )
                return
        colors = self._colors
        if self.heatmap_cbar is not None:
            self.heatmap_cbar.remove()
            self.heatmap_cbar = None

        view = self._active_heatmap_view()
        if view is None:
            self._draw_plot_placeholder(
                f"Run the {self._active_left_tab_label()} compute to populate plot."
            )
            return

        selected = self._get_selected_metric_grid(view)
        if selected is None:
            self._draw_plot_placeholder("Select a valid metric.")
            return
        metric_label, metric_key, z = selected
        try:
            cmin, cmax = self._get_color_limits()
        except Exception as exc:
            messagebox.showerror("Colorbar", str(exc))
            return

        x_vals = view.x_vals
        freqs = view.freqs
        self.ax_heatmap.clear()
        cmap = "magma" if "[span]" in metric_label else ("twilight" if "phase" in metric_key else "viridis")
        im = self.ax_heatmap.imshow(
            z,
            origin="lower",
            aspect="auto",
            extent=(x_vals[0], x_vals[-1], freqs[0], freqs[-1]),
            cmap=cmap,
            vmin=cmin,
            vmax=cmax,
        )
        self.ax_heatmap.set_title(f"{metric_label} vs {view.x_name}/Frequency")
        self.ax_heatmap.set_xlabel(view.x_label)
        self.ax_heatmap.set_ylabel("Frequency (GHz)")
        self._style_plot_axis(self.ax_heatmap)
        self.ax_heatmap.grid(False)
        self.heatmap_cbar = self.fig.colorbar(im, ax=self.ax_heatmap)
        if cmin is None or cmax is None:
            self.heatmap_cbar.set_label(metric_label)
        else:
            self.heatmap_cbar.set_label(f"{metric_label} [{cmin:g}, {cmax:g}]")
        style_colorbar(self.heatmap_cbar, colors)

        self._update_slice_plots(view, z, metric_label)
        self.canvas.draw_idle()

    def _compute_heatmap_data(
        self,
        loaded_layers: list[LoadedLayer],
        wave_pol: str,
        angles: list[float],
        freqs: list[float] | None = None,
        thickness_scale: float = 1.0,
        eps_scale: float = 1.0,
        mu_scale: float = 1.0,
    ) -> dict[str, list[list[float]] | list[float]]:
        if freqs is None:
            f_start = float(self.f_start_var.get().strip())
            f_stop = float(self.f_stop_var.get().strip())
            f_step = float(self.f_step_var.get().strip())
            freqs = make_frequency_sweep(f_start, f_stop, f_step)

        for i, layer in enumerate(loaded_layers, start=1):
            if layer.is_sheet:
                continue
            validate_sweep_coverage(freqs, layer.table_0deg, f"layer {i} 0 deg/isotropic")
            if layer.anisotropic:
                if layer.table_90deg is None:
                    raise ValueError(f"Layer {i}: anisotropic layer is missing a 90 deg table.")
                validate_sweep_coverage(freqs, layer.table_90deg, f"layer {i} 90 deg")

        if NUMPY_AVAILABLE:
            prepared_properties = prepare_layer_properties_many(freqs, loaded_layers)
            grids = {k: np.zeros((len(freqs), len(angles)), dtype=float) for k in HEATMAP_METRIC_KEYS}
            for j, a in enumerate(angles):
                col = compute_angle_metrics_many(
                    freqs,
                    a,
                    loaded_layers,
                    wave_pol,
                    thickness_scale=thickness_scale,
                    eps_scale=eps_scale,
                    mu_scale=mu_scale,
                    prepared_properties=prepared_properties,
                )
                for key in HEATMAP_METRIC_KEYS:
                    grids[key][:, j] = np.asarray(col[key], dtype=float)
            metric_grids = {k: grids[k].tolist() for k in HEATMAP_METRIC_KEYS}
        else:
            metric_grids = {k: [] for k in HEATMAP_METRIC_KEYS}
            for f_ghz in freqs:
                row = compute_angle_metrics(
                    f_ghz,
                    angles[0],
                    loaded_layers,
                    wave_pol,
                    thickness_scale=thickness_scale,
                    eps_scale=eps_scale,
                    mu_scale=mu_scale,
                )
                for key in HEATMAP_METRIC_KEYS:
                    metric_grids[key].append([row[key]])
            for j in range(1, len(angles)):
                for i, f_ghz in enumerate(freqs):
                    m = compute_angle_metrics(
                        f_ghz,
                        angles[j],
                        loaded_layers,
                        wave_pol,
                        thickness_scale=thickness_scale,
                        eps_scale=eps_scale,
                        mu_scale=mu_scale,
                    )
                    for key in HEATMAP_METRIC_KEYS:
                        metric_grids[key][i].append(m[key])

        return {
            "angle_deg": angles,
            "freq_ghz": freqs,
            **metric_grids,
        }

    def _compute_thickness_data(
        self,
        loaded_layers: list[LoadedLayer],
        layer_idx: int,
        thicknesses_in: list[float],
        wave_pol: str,
        angle_deg: float,
        freqs: list[float],
        thickness_scale: float = 1.0,
        eps_scale: float = 1.0,
        mu_scale: float = 1.0,
    ) -> dict[str, list[list[float]] | list[float]]:
        """Metric grids indexed [frequency][thickness] for one layer swept over
        ``thicknesses_in`` at a fixed incidence angle. Material tables are read
        once by the caller and shared across every thickness."""
        for i, layer in enumerate(loaded_layers, start=1):
            if layer.is_sheet:
                continue
            validate_sweep_coverage(freqs, layer.table_0deg, f"layer {i} 0 deg/isotropic")
            if layer.anisotropic:
                if layer.table_90deg is None:
                    raise ValueError(f"Layer {i}: anisotropic layer is missing a 90 deg table.")
                validate_sweep_coverage(freqs, layer.table_90deg, f"layer {i} 90 deg")

        swept = loaded_layers[layer_idx]

        def stack_for(thickness_in: float) -> list[LoadedLayer]:
            trial = list(loaded_layers)
            trial[layer_idx] = LoadedLayer(
                thickness_m=thickness_in * INCH_TO_M,
                anisotropic=swept.anisotropic,
                polarization_deg=swept.polarization_deg,
                table_0deg=swept.table_0deg,
                table_90deg=swept.table_90deg,
                is_sheet=swept.is_sheet,
                sheet_resistance=swept.sheet_resistance,
            )
            return trial

        if NUMPY_AVAILABLE:
            prepared_properties = prepare_layer_properties_many(freqs, loaded_layers)
            prepared_wave_terms = prepare_layer_wave_terms_many(
                freqs,
                angle_deg,
                loaded_layers,
                wave_pol,
                eps_scale=eps_scale,
                mu_scale=mu_scale,
                prepared_properties=prepared_properties,
            )
            grids = {
                k: np.zeros((len(freqs), len(thicknesses_in)), dtype=float)
                for k in HEATMAP_METRIC_KEYS
            }
            for j, t_in in enumerate(thicknesses_in):
                col = compute_angle_metrics_many(
                    freqs,
                    angle_deg,
                    stack_for(t_in),
                    wave_pol,
                    thickness_scale=thickness_scale,
                    eps_scale=eps_scale,
                    mu_scale=mu_scale,
                    prepared_wave_terms=prepared_wave_terms,
                )
                for key in HEATMAP_METRIC_KEYS:
                    grids[key][:, j] = np.asarray(col[key], dtype=float)
            metric_grids = {k: grids[k].tolist() for k in HEATMAP_METRIC_KEYS}
        else:
            metric_grids = {k: [[] for _ in freqs] for k in HEATMAP_METRIC_KEYS}
            for t_in in thicknesses_in:
                col = compute_angle_metrics_many(
                    freqs,
                    angle_deg,
                    stack_for(t_in),
                    wave_pol,
                    thickness_scale=thickness_scale,
                    eps_scale=eps_scale,
                    mu_scale=mu_scale,
                )
                for key in HEATMAP_METRIC_KEYS:
                    for i in range(len(freqs)):
                        metric_grids[key][i].append(col[key][i])

        return {
            "thickness_in": thicknesses_in,
            "freq_ghz": freqs,
            **metric_grids,
        }

    def _compute_frequency_mode(
        self,
        output_path: Path,
        include_header: bool,
        loaded_layers: list[LoadedLayer],
        backing: str,
        uncertainty: UncertaintyConfig,
        sweep: list[float],
        wave_pol: str,
    ) -> tuple[int, str]:
        for i, layer in enumerate(loaded_layers, start=1):
            if layer.is_sheet:
                continue
            validate_sweep_coverage(sweep, layer.table_0deg, f"layer {i} 0 deg/isotropic")
            if layer.anisotropic:
                if layer.table_90deg is None:
                    raise ValueError(f"Layer {i}: anisotropic layer is missing a 90 deg table.")
                validate_sweep_coverage(sweep, layer.table_90deg, f"layer {i} 90 deg")

        z_nom = compute_stack_impedance_many(sweep, loaded_layers, backing)
        scales = build_uncertainty_scales(uncertainty)
        envelope_enabled = uncertainty.enabled and len(scales) > 1

        # The nominal file always retains the exact three-column schema used
        # by both RCS solvers. Uncertainty bounds belong in a separate analysis
        # report and must never make the solver input incompatible.
        nominal_rows = [
            (f_ghz, z.real, z.imag) for f_ghz, z in zip(sweep, z_nom)
        ]
        uncertainty_path: Path | None = None
        uncertainty_rows = None
        if envelope_enabled:
            zr_nom = [z.real for z in z_nom]
            zi_nom = [z.imag for z in z_nom]
            zr_min = zr_nom.copy()
            zr_max = zr_nom.copy()
            zi_min = zi_nom.copy()
            zi_max = zi_nom.copy()
            for t_scale, e_scale, m_scale in scales:
                if is_nominal_scale(t_scale, e_scale, m_scale):
                    continue
                z_s = compute_stack_impedance_many(
                    sweep,
                    loaded_layers,
                    backing,
                    thickness_scale=t_scale,
                    eps_scale=e_scale,
                    mu_scale=m_scale,
                )
                for i, z in enumerate(z_s):
                    zr = z.real
                    zi = z.imag
                    zr_min[i] = min(zr_min[i], zr)
                    zr_max[i] = max(zr_max[i], zr)
                    zi_min[i] = min(zi_min[i], zi)
                    zi_max[i] = max(zi_max[i], zi)

            uncertainty_path = uncertainty_report_path(output_path)
            uncertainty_rows = [
                (
                    f_ghz,
                    zr_nom[i],
                    zi_nom[i],
                    zr_min[i],
                    zr_max[i],
                    zi_min[i],
                    zi_max[i],
                )
                for i, f_ghz in enumerate(sweep)
            ]
        write_impedance_bundle(
            output_path,
            nominal_rows,
            include_header,
            uncertainty_path,
            uncertainty_rows,
        )

        summary = self._summarize_frequency_run(
            sweep,
            loaded_layers,
            wave_pol,
            envelope_enabled,
            backing,
        )
        if uncertainty_path is not None:
            summary += f"\nUncertainty report: {uncertainty_path}"
        return len(sweep), summary

    def _compute_angle_mode(
        self,
        output_path: Path,
        include_header: bool,
        loaded_layers: list[LoadedLayer],
        uncertainty: UncertaintyConfig,
        angles: list[float],
        freqs: list[float],
        wave_pol: str,
    ) -> tuple[
        int,
        dict[str, list[list[float]] | list[float]],
        dict[str, list[list[float]]] | None,
        dict[str, list[list[float]]] | None,
        str,
    ]:
        out = self._compute_heatmap_data(loaded_layers, wave_pol, angles, freqs=freqs)

        scales = build_uncertainty_scales(uncertainty)
        envelope_enabled = uncertainty.enabled and len(scales) > 1
        envelope_min: dict[str, list[list[float]]] | None = None
        envelope_max: dict[str, list[list[float]]] | None = None
        if envelope_enabled:
            if NUMPY_AVAILABLE:
                envelope_min = {
                    key: np.asarray(out[key], dtype=float)
                    for key in HEATMAP_METRIC_KEYS
                }
                envelope_max = {
                    key: np.asarray(out[key], dtype=float)
                    for key in HEATMAP_METRIC_KEYS
                }
                for t_scale, e_scale, m_scale in scales:
                    if is_nominal_scale(t_scale, e_scale, m_scale):
                        continue
                    s_out = self._compute_heatmap_data(
                        loaded_layers,
                        wave_pol,
                        angles,
                        freqs=freqs,
                        thickness_scale=t_scale,
                        eps_scale=e_scale,
                        mu_scale=m_scale,
                    )
                    for key in HEATMAP_METRIC_KEYS:
                        arr = np.asarray(s_out[key], dtype=float)
                        if key in PHASE_METRIC_KEYS:
                            nominal = np.asarray(out[key], dtype=float)
                            arr = nominal + (arr - nominal + 180.0) % 360.0 - 180.0
                        envelope_min[key] = np.minimum(envelope_min[key], arr)
                        envelope_max[key] = np.maximum(envelope_max[key], arr)
                envelope_min = {key: envelope_min[key].tolist() for key in HEATMAP_METRIC_KEYS}
                envelope_max = {key: envelope_max[key].tolist() for key in HEATMAP_METRIC_KEYS}
            else:
                envelope_min = {
                    key: [[v for v in row] for row in out[key]]
                    for key in HEATMAP_METRIC_KEYS
                }
                envelope_max = {
                    key: [[v for v in row] for row in out[key]]
                    for key in HEATMAP_METRIC_KEYS
                }
                for t_scale, e_scale, m_scale in scales:
                    if is_nominal_scale(t_scale, e_scale, m_scale):
                        continue
                    s_out = self._compute_heatmap_data(
                        loaded_layers,
                        wave_pol,
                        angles,
                        freqs=freqs,
                        thickness_scale=t_scale,
                        eps_scale=e_scale,
                        mu_scale=m_scale,
                    )
                    for key in HEATMAP_METRIC_KEYS:
                        for i in range(len(out["freq_ghz"])):
                            for j in range(len(out["angle_deg"])):
                                val = s_out[key][i][j]
                                if key in PHASE_METRIC_KEYS:
                                    val = align_phase_degrees(
                                        val, out[key][i][j]
                                    )
                                envelope_min[key][i][j] = min(envelope_min[key][i][j], val)
                                envelope_max[key][i][j] = max(envelope_max[key][i][j], val)

        freq = out["freq_ghz"]
        ang = out["angle_deg"]
        metal_loss = out["metal_loss_db"]
        metal_phase = out["metal_phase_deg"]
        metal_abs = out["metal_absorption_db"]
        air_loss = out["air_loss_db"]
        air_phase = out["air_phase_deg"]
        air_abs = out["air_absorption_db"]
        insertion_loss = out["insertion_loss_db"]
        insertion_phase = out["insertion_phase_deg"]

        with _atomic_text_file(output_path) as f:
            if include_header:
                if envelope_enabled:
                    f.write(
                        "frequency_hz,angle_deg,"
                        "metal_loss_db,metal_loss_db_min,metal_loss_db_max,"
                        "metal_phase_deg,metal_phase_deg_min,metal_phase_deg_max,"
                        "metal_absorption_db,metal_absorption_db_min,metal_absorption_db_max,"
                        "air_loss_db,air_loss_db_min,air_loss_db_max,"
                        "air_phase_deg,air_phase_deg_min,air_phase_deg_max,"
                        "air_absorption_db,air_absorption_db_min,air_absorption_db_max,"
                        "insertion_loss_db,insertion_loss_db_min,insertion_loss_db_max,"
                        "insertion_phase_deg,insertion_phase_deg_min,insertion_phase_deg_max\n"
                    )
                else:
                    f.write(
                        "frequency_hz,angle_deg,metal_loss_db,metal_phase_deg,metal_absorption_db,"
                        "air_loss_db,air_phase_deg,air_absorption_db,"
                        "insertion_loss_db,insertion_phase_deg\n"
                    )
            for i, f_ghz in enumerate(freq):
                for j, a in enumerate(ang):
                    if envelope_enabled:
                        if envelope_min is None or envelope_max is None:
                            raise ValueError("Internal error: uncertainty envelopes are unavailable.")
                        f.write(
                            f"{f_ghz * HZ_PER_GHZ:.12g},{a:.12g},"
                            f"{metal_loss[i][j]:.12g},"
                            f"{envelope_min['metal_loss_db'][i][j]:.12g},{envelope_max['metal_loss_db'][i][j]:.12g},"
                            f"{metal_phase[i][j]:.12g},"
                            f"{envelope_min['metal_phase_deg'][i][j]:.12g},{envelope_max['metal_phase_deg'][i][j]:.12g},"
                            f"{metal_abs[i][j]:.12g},"
                            f"{envelope_min['metal_absorption_db'][i][j]:.12g},{envelope_max['metal_absorption_db'][i][j]:.12g},"
                            f"{air_loss[i][j]:.12g},"
                            f"{envelope_min['air_loss_db'][i][j]:.12g},{envelope_max['air_loss_db'][i][j]:.12g},"
                            f"{air_phase[i][j]:.12g},"
                            f"{envelope_min['air_phase_deg'][i][j]:.12g},{envelope_max['air_phase_deg'][i][j]:.12g},"
                            f"{air_abs[i][j]:.12g},"
                            f"{envelope_min['air_absorption_db'][i][j]:.12g},{envelope_max['air_absorption_db'][i][j]:.12g},"
                            f"{insertion_loss[i][j]:.12g},"
                            f"{envelope_min['insertion_loss_db'][i][j]:.12g},{envelope_max['insertion_loss_db'][i][j]:.12g},"
                            f"{insertion_phase[i][j]:.12g},"
                            f"{envelope_min['insertion_phase_deg'][i][j]:.12g},{envelope_max['insertion_phase_deg'][i][j]:.12g}\n"
                        )
                    else:
                        f.write(
                            f"{f_ghz * HZ_PER_GHZ:.12g},{a:.12g},"
                            f"{metal_loss[i][j]:.12g},{metal_phase[i][j]:.12g},{metal_abs[i][j]:.12g},"
                            f"{air_loss[i][j]:.12g},{air_phase[i][j]:.12g},{air_abs[i][j]:.12g},"
                            f"{insertion_loss[i][j]:.12g},{insertion_phase[i][j]:.12g}\n"
                        )

        summary = self._summarize_angle_run(out, wave_pol, envelope_enabled)
        return len(freq) * len(ang), out, envelope_min, envelope_max, summary

    def _compute_thickness_mode(
        self,
        output_path: Path,
        include_header: bool,
        loaded_layers: list[LoadedLayer],
        layer_idx: int,
        uncertainty: UncertaintyConfig,
        thicknesses_in: list[float],
        freqs: list[float],
        wave_pol: str,
        angle_deg: float,
    ) -> tuple[
        int,
        dict[str, list[list[float]] | list[float]],
        dict[str, list[list[float]]] | None,
        dict[str, list[list[float]]] | None,
        str,
    ]:
        out = self._compute_thickness_data(
            loaded_layers, layer_idx, thicknesses_in, wave_pol, angle_deg, freqs
        )

        scales = build_uncertainty_scales(uncertainty)
        envelope_enabled = uncertainty.enabled and len(scales) > 1
        envelope_min: dict[str, list[list[float]]] | None = None
        envelope_max: dict[str, list[list[float]]] | None = None
        if envelope_enabled:
            envelope_min = {
                key: [list(row) for row in out[key]] for key in HEATMAP_METRIC_KEYS
            }
            envelope_max = {
                key: [list(row) for row in out[key]] for key in HEATMAP_METRIC_KEYS
            }
            for t_scale, e_scale, m_scale in scales:
                if is_nominal_scale(t_scale, e_scale, m_scale):
                    continue
                s_out = self._compute_thickness_data(
                    loaded_layers,
                    layer_idx,
                    thicknesses_in,
                    wave_pol,
                    angle_deg,
                    freqs,
                    thickness_scale=t_scale,
                    eps_scale=e_scale,
                    mu_scale=m_scale,
                )
                for key in HEATMAP_METRIC_KEYS:
                    grid = s_out[key]
                    for i in range(len(freqs)):
                        for j in range(len(thicknesses_in)):
                            val = grid[i][j]
                            if key in PHASE_METRIC_KEYS:
                                val = align_phase_degrees(
                                    val, out[key][i][j]
                                )
                            if val < envelope_min[key][i][j]:
                                envelope_min[key][i][j] = val
                            if val > envelope_max[key][i][j]:
                                envelope_max[key][i][j] = val

        with _atomic_text_file(output_path) as f:
            if include_header:
                cols = ["frequency_hz", "thickness_in"]
                for key in HEATMAP_METRIC_KEYS:
                    cols.append(key)
                    if envelope_enabled:
                        cols.extend((f"{key}_min", f"{key}_max"))
                f.write(",".join(cols) + "\n")
            for i, f_ghz in enumerate(freqs):
                for j, t_in in enumerate(thicknesses_in):
                    vals = [f_ghz * HZ_PER_GHZ, t_in]
                    for key in HEATMAP_METRIC_KEYS:
                        vals.append(out[key][i][j])
                        if envelope_enabled:
                            if envelope_min is None or envelope_max is None:
                                raise ValueError(
                                    "Internal error: uncertainty envelopes are unavailable."
                                )
                            vals.append(envelope_min[key][i][j])
                            vals.append(envelope_max[key][i][j])
                    f.write(",".join(f"{v:.12g}" for v in vals) + "\n")

        summary = self._summarize_thickness_run(out, wave_pol, angle_deg, envelope_enabled)
        return len(freqs) * len(thicknesses_in), out, envelope_min, envelope_max, summary

    def _read_inverse_uncertainty_config(self) -> UncertaintyConfig:
        if not self.inv_uncertainty_var.get():
            return UncertaintyConfig(enabled=False, thickness_pct=0.0, eps_pct=0.0, mu_pct=0.0)

        t_pct = float(self.inv_unc_t_pct_var.get().strip())
        eps_pct = float(self.inv_unc_eps_pct_var.get().strip())
        mu_pct = float(self.inv_unc_mu_pct_var.get().strip())
        if not all(math.isfinite(value) for value in (t_pct, eps_pct, mu_pct)):
            raise ValueError("Inverse-design uncertainty percentages must be finite.")
        if t_pct < 0 or eps_pct < 0 or mu_pct < 0:
            raise ValueError("Inverse-design uncertainty percentages must be >= 0.")
        if t_pct >= 100 or eps_pct >= 100 or mu_pct >= 100:
            raise ValueError(
                "Inverse-design uncertainty percentages must be < 100."
            )
        return UncertaintyConfig(enabled=True, thickness_pct=t_pct, eps_pct=eps_pct, mu_pct=mu_pct)

    def _parse_inverse_discrete_freqs(self, text: str) -> list[float]:
        tokens = (
            text.replace(",", " ")
            .replace(";", " ")
            .replace("\n", " ")
            .split()
        )
        if not tokens:
            raise ValueError("Enter one or more discrete frequencies in GHz (for example: 8.2, 9.5, 10.0).")
        values: list[float] = []
        for token in tokens:
            value = float(token)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "Discrete frequencies must be finite and > 0 GHz."
                )
            values.append(value)
        unique_sorted = sorted(set(values))
        if not unique_sorted:
            raise ValueError("No valid discrete frequencies were provided.")
        return unique_sorted

    def _score_inverse_candidate(
        self,
        target_freqs: list[float],
        target_angles: list[float],
        candidate_layers: list[LoadedLayer],
        wave_pol: str,
        scales: list[tuple[float, float, float]],
        score_mode: str,
        prepared_wave_terms: dict[
            tuple[float, float, float],
            list[tuple["np.ndarray", "np.ndarray"] | None],
        ] | None = None,
    ) -> tuple[float, float, float, float, float]:
        corner_means: list[float] = []
        nominal_mean: float | None = None
        for t_scale, e_scale, m_scale in scales:
            values: list[float] = []
            for angle_deg in target_angles:
                prepared = (
                    prepared_wave_terms.get((angle_deg, e_scale, m_scale))
                    if prepared_wave_terms is not None
                    else None
                )
                metrics = compute_angle_metrics_many(
                    target_freqs,
                    angle_deg,
                    candidate_layers,
                    wave_pol,
                    thickness_scale=t_scale,
                    eps_scale=e_scale,
                    mu_scale=m_scale,
                    prepared_wave_terms=prepared,
                )
                values.extend(metrics["metal_loss_db"])
            mean_db, _mn, _mx = self._stats(values)
            corner_means.append(mean_db)
            if abs(t_scale - 1.0) < 1e-12 and abs(e_scale - 1.0) < 1e-12 and abs(m_scale - 1.0) < 1e-12:
                nominal_mean = mean_db

        if not corner_means:
            raise ValueError("No corner scores computed for inverse candidate.")
        if nominal_mean is None:
            nominal_mean = corner_means[0]

        worst_mean = max(corner_means)
        avg_mean = sum(corner_means) / len(corner_means)
        best_mean = min(corner_means)
        score_db = worst_mean if "worst-case" in score_mode.lower() else avg_mean
        return score_db, nominal_mean, worst_mean, avg_mean, best_mean

    def _refresh_inverse_results_list(self) -> None:
        if self.inv_results_list is None:
            return
        self.inv_results_list.clear()
        for i, c in enumerate(self.inverse_candidates, start=1):
            geom_parts = []
            for li in range(len(c.thickness_in)):
                if li < len(c.sheet_resistance_ohm) and c.sheet_resistance_ohm[li] > 0:
                    geom_parts.append(f"{c.sheet_resistance_ohm[li]:g}Ω")
                else:
                    geom_parts.append(f"{c.thickness_in[li]:.4g}in")
            geom_text = ", ".join(geom_parts)
            m_text = ", ".join(Path(p).name for p in c.material_files if p)
            line = (
                f"{i:02d}: score={c.score_db:.3f} dB | nom={c.nominal_mean_db:.3f} | "
                f"worst={c.worst_mean_db:.3f} | avg={c.avg_mean_db:.3f} | "
                f"geom=[{geom_text}] | m=[{m_text}]"
            )
            self.inv_results_list.addItem(line)
        self._update_plot()

    def _apply_inverse_candidate(self) -> None:
        try:
            if not self.inverse_candidates or self.inv_results_list is None:
                messagebox.showwarning("Inverse Design", "Run inverse design first.")
                return
            row = self.inv_results_list.currentRow()
            if row < 0:
                messagebox.showwarning("Inverse Design", "Select a candidate to apply.")
                return
            idx = int(row)
            if idx < 0 or idx >= len(self.inverse_candidates):
                messagebox.showwarning("Inverse Design", "Selected candidate is out of range.")
                return

            cand = self.inverse_candidates[idx]
            if (
                len(cand.thickness_in) != len(self.layers)
                or len(cand.material_files) != len(self.layers)
                or len(cand.sheet_resistance_ohm) != len(self.layers)
            ):
                raise ValueError("Layer count changed since inverse design run. Re-run inverse design.")

            for i, layer in enumerate(self.layers):
                if layer.is_sheet:
                    if cand.sheet_resistance_ohm[i] > 0:
                        layer.sheet_resistance = cand.sheet_resistance_ohm[i]
                    continue
                layer.thickness_in = cand.thickness_in[i]
                if not layer.anisotropic:
                    layer.file_0deg = cand.material_files[i]
            self._refresh_layers()

            msg = (
                f"Applied inverse candidate #{idx + 1}.\n"
                f"Score: {cand.score_db:.3f} dB | Nominal: {cand.nominal_mean_db:.3f} dB | "
                f"Worst-case: {cand.worst_mean_db:.3f} dB"
            )
            messagebox.showinfo("Inverse Design", msg)
        except Exception as exc:
            messagebox.showerror("Inverse Design Error", str(exc))

    def _run_inverse_design(self) -> None:
        try:
            if not self.layers:
                raise ValueError("Add at least one layer before inverse design.")

            layer_snapshot = self._snapshot_layers()
            skiprows = 0
            wave_pol = normalize_wave_polarization(self.inv_wave_pol_var.get())
            freq_mode = self.inv_freq_mode_var.get().strip().lower()
            if freq_mode.startswith("discrete"):
                target_freqs = self._parse_inverse_discrete_freqs(self.inv_freq_list_var.get())
                target_freq_desc = "Discrete GHz: " + ", ".join(f"{v:g}" for v in target_freqs)
            else:
                f_start = float(self.inv_target_start_var.get().strip())
                f_stop = float(self.inv_target_stop_var.get().strip())
                f_step = float(self.inv_target_step_var.get().strip())
                target_freqs = make_frequency_sweep(f_start, f_stop, f_step)
                target_freq_desc = f"Band GHz: {f_start:g}-{f_stop:g} (step {f_step:g})"
            a_start = float(self.inv_angle_start_var.get().strip())
            a_stop = float(self.inv_angle_stop_var.get().strip())
            a_start = validate_incidence_angle(a_start)
            a_stop = validate_incidence_angle(a_stop)
            if a_stop < a_start:
                raise ValueError("Inverse-design angle stop must be >= start.")
            if abs(a_stop - a_start) <= 1e-12:
                target_angles = [a_start]
            else:
                a_step = float(self.inv_angle_step_var.get().strip())
                target_angles = make_sweep(a_start, a_stop, a_step)

            max_evals = int(self.inv_max_evals_var.get().strip())
            top_n = int(self.inv_top_n_var.get().strip())
            if max_evals <= 0:
                raise ValueError("Inverse-design Max evals must be >= 1.")
            if top_n <= 0:
                raise ValueError("Inverse-design Top N must be >= 1.")
            score_mode = self.inv_score_mode_var.get().strip()
            uncertainty_cfg = self._read_inverse_uncertainty_config()
            refine_top_candidates = bool(self.inv_refine_var.get())
            if refine_top_candidates and (not NUMPY_AVAILABLE or not SCIPY_AVAILABLE):
                raise ValueError(
                    "Local inverse-design refinement requires NumPy and SciPy."
                )
            seed_text = self.inv_seed_var.get().strip()
            search_seed: int | None = int(seed_text) if seed_text else None
        except Exception as exc:
            messagebox.showerror("Inverse Design Error", str(exc))
            return

        def worker() -> tuple[list[InverseCandidate], str, list[float], list[list[list[float]]]]:
            scales = build_uncertainty_scales(uncertainty_cfg)

            table_cache: dict[str, MaterialTable] = {}

            def get_table(path_str: str) -> MaterialTable:
                key = str(Path(path_str))
                if key not in table_cache:
                    table_cache[key] = read_material_table(Path(key), skiprows)
                return table_cache[key]

            def prepare_material_combo(
                chosen_files: list[str],
            ) -> tuple[bool, list[MaterialTable | None], list[MaterialTable | None]]:
                tables_0_local: list[MaterialTable | None] = []
                tables_90_local: list[MaterialTable | None] = []
                for i, layer in enumerate(layer_snapshot, start=1):
                    if layer.is_sheet:
                        tables_0_local.append(None)
                        tables_90_local.append(None)
                        continue
                    table_0 = get_table(chosen_files[i - 1])
                    try:
                        validate_sweep_coverage(
                            target_freqs, table_0, f"inverse layer {i} 0deg/isotropic"
                        )
                    except Exception:
                        return False, [], []
                    table_90: MaterialTable | None = None
                    if layer.anisotropic:
                        table_90 = get_table(layer.file_90deg)
                        try:
                            validate_sweep_coverage(
                                target_freqs, table_90, f"inverse layer {i} 90deg"
                            )
                        except Exception:
                            return False, [], []
                    tables_0_local.append(table_0)
                    tables_90_local.append(table_90)
                return True, tables_0_local, tables_90_local

            # Materials are fixed during the search, so the validated tables
            # (tables_0_mc / tables_90_mc, assigned below) are reused for every
            # candidate. Each candidate is fully described by its per-layer
            # thicknesses (bulk) and resistances (sheets).
            def build_loaded_layers(
                thicknesses: list[float], resistances: list[float]
            ) -> list[LoadedLayer]:
                out: list[LoadedLayer] = []
                for i, layer in enumerate(layer_snapshot):
                    if layer.is_sheet:
                        out.append(
                            LoadedLayer(
                                thickness_m=0.0,
                                anisotropic=False,
                                polarization_deg=0.0,
                                table_0deg=None,
                                table_90deg=None,
                                is_sheet=True,
                                sheet_resistance=resistances[i],
                            )
                        )
                    else:
                        out.append(
                            LoadedLayer(
                                thickness_m=thicknesses[i] * INCH_TO_M,
                                anisotropic=layer.anisotropic,
                                polarization_deg=layer.polarization_deg,
                                table_0deg=tables_0_mc[i],
                                table_90deg=tables_90_mc[i],
                            )
                        )
                return out

            def score_config(
                thicknesses: list[float], resistances: list[float]
            ) -> tuple[float, float, float, float, float]:
                return self._score_inverse_candidate(
                    target_freqs,
                    target_angles,
                    build_loaded_layers(thicknesses, resistances),
                    wave_pol,
                    scales,
                    score_mode,
                    prepared_inverse_wave_terms,
                )

            top_candidates: list[InverseCandidate] = []
            eval_count = 0
            refine_evals = 0
            prepared_inverse_wave_terms: dict[
                tuple[float, float, float],
                list[tuple[np.ndarray, np.ndarray] | None],
            ] = {}

            # Monte Carlo: one fixed material per layer; validate coverage once.
            chosen_files = [
                "" if layer.is_sheet else layer.file_0deg for layer in layer_snapshot
            ]
            coverage_ok, tables_0_mc, tables_90_mc = prepare_material_combo(chosen_files)
            if not coverage_ok:
                raise ValueError(
                    "Selected layer materials do not cover the inverse-design "
                    "frequency target. Check each layer's property file range."
                )

            # Build the search-variable list. Each entry is
            # (layer_index, kind, lo, hi, accuracy) where kind is "t" (bulk-layer
            # thickness, in) or "rs" (sheet resistance, ohms). Bulk layers always
            # require thickness bounds; sheets are searched only when R min/max
            # are set, otherwise they stay fixed at their nominal resistance.
            search_dims: list[tuple[int, str, float, float, float | None]] = []
            base_thick: list[float] = [0.0] * len(layer_snapshot)
            base_rs: list[float] = [0.0] * len(layer_snapshot)
            for layer_idx, layer in enumerate(layer_snapshot):
                human_idx = layer_idx + 1
                if layer.is_sheet:
                    rs0 = layer.sheet_resistance
                    if layer.inv_rs_min is not None or layer.inv_rs_max is not None:
                        if layer.inv_rs_min is None or layer.inv_rs_max is None:
                            raise ValueError(
                                f"Layer {human_idx}: set both sheet R min and max, "
                                "or neither."
                            )
                        if layer.inv_rs_min <= 0 or layer.inv_rs_max <= 0:
                            raise ValueError(
                                f"Layer {human_idx}: sheet resistance bounds must be > 0."
                            )
                        if layer.inv_rs_max < layer.inv_rs_min:
                            raise ValueError(
                                f"Layer {human_idx}: sheet R max must be >= min."
                            )
                        rs0 = layer.inv_rs_min
                        if layer.inv_rs_max > layer.inv_rs_min:
                            search_dims.append(
                                (
                                    layer_idx,
                                    "rs",
                                    layer.inv_rs_min,
                                    layer.inv_rs_max,
                                    layer.inv_rs_accuracy,
                                )
                            )
                    base_rs[layer_idx] = rs0
                    continue

                if layer.inv_t_min_in is None or layer.inv_t_max_in is None:
                    raise ValueError(
                        f"Layer {human_idx}: set inverse-design t_min and t_max "
                        "on this layer before running Monte Carlo."
                    )
                if layer.inv_t_min_in <= 0 or layer.inv_t_max_in <= 0:
                    raise ValueError(f"Layer {human_idx}: thickness bounds must be > 0.")
                if layer.inv_t_max_in < layer.inv_t_min_in:
                    raise ValueError(f"Layer {human_idx}: thickness max must be >= min.")
                base_thick[layer_idx] = layer.inv_t_min_in
                if layer.inv_t_max_in > layer.inv_t_min_in:
                    search_dims.append(
                        (
                            layer_idx,
                            "t",
                            layer.inv_t_min_in,
                            layer.inv_t_max_in,
                            layer.inv_t_accuracy_in,
                        )
                    )

            if NUMPY_AVAILABLE:
                reference_layers = build_loaded_layers(base_thick, base_rs)
                prepared_properties = prepare_layer_properties_many(
                    target_freqs, reference_layers
                )
                for _t_scale, e_scale, m_scale in scales:
                    for angle_deg in target_angles:
                        key = (angle_deg, e_scale, m_scale)
                        if key not in prepared_inverse_wave_terms:
                            prepared_inverse_wave_terms[key] = (
                                prepare_layer_wave_terms_many(
                                    target_freqs,
                                    angle_deg,
                                    reference_layers,
                                    wave_pol,
                                    eps_scale=e_scale,
                                    mu_scale=m_scale,
                                    prepared_properties=prepared_properties,
                                )
                            )

            mc_rng = random.Random(search_seed)
            n_samples = max_evals if search_dims else 1

            for _ in range(n_samples):
                trial_t = list(base_thick)
                trial_rs = list(base_rs)
                for idx, kind, lo, hi, acc in search_dims:
                    val = snap_to_increment(mc_rng.uniform(lo, hi), acc, lo, hi)
                    if kind == "t":
                        trial_t[idx] = val
                    else:
                        trial_rs[idx] = val
                eval_count += 1
                score_db, nominal_mean, worst_mean, avg_mean, best_mean = score_config(
                    trial_t, trial_rs
                )
                top_candidates.append(
                    InverseCandidate(
                        score_db=score_db,
                        nominal_mean_db=nominal_mean,
                        worst_mean_db=worst_mean,
                        avg_mean_db=avg_mean,
                        best_mean_db=best_mean,
                        thickness_in=[float(v) for v in trial_t],
                        material_files=chosen_files[:],
                        sheet_resistance_ohm=[float(v) for v in trial_rs],
                    )
                )
                top_candidates.sort(key=lambda c: c.score_db, reverse=False)
                if len(top_candidates) > top_n:
                    del top_candidates[top_n:]

            if not top_candidates:
                raise ValueError("Inverse design found no valid candidates in the target region.")

            if refine_top_candidates and search_dims:
                # Refine every search variable (thickness and sheet resistance)
                # in normalized [0, 1] coordinates so thickness (~0.1 in) and
                # resistance (~100 ohm) carry equal weight in the simplex. The
                # accuracy increment is applied at evaluation, so the optimizer
                # only ever sees on-grid, in-bounds configurations.
                def unpack(
                    x_free: "np.ndarray", cand: InverseCandidate
                ) -> tuple[list[float], list[float]]:
                    trial_t = list(cand.thickness_in)
                    trial_rs = list(cand.sheet_resistance_ohm)
                    for di, (idx, kind, lo, hi, acc) in enumerate(search_dims):
                        val = snap_to_increment(lo + float(x_free[di]) * (hi - lo), acc, lo, hi)
                        if kind == "t":
                            trial_t[idx] = val
                        else:
                            trial_rs[idx] = val
                    return trial_t, trial_rs

                bounds_arr = [(0.0, 1.0)] * len(search_dims)
                refined: list[InverseCandidate] = []

                for cand in top_candidates:
                    x0 = np.array(
                        [
                            (
                                (cand.thickness_in[idx] if kind == "t"
                                 else cand.sheet_resistance_ohm[idx]) - lo
                            )
                            / (hi - lo)
                            for idx, kind, lo, hi, _acc in search_dims
                        ],
                        dtype=float,
                    )
                    x0 = np.clip(x0, 0.0, 1.0)

                    def objective(
                        x_free: "np.ndarray", _cand: InverseCandidate = cand
                    ) -> float:
                        nonlocal refine_evals
                        trial_t, trial_rs = unpack(x_free, _cand)
                        refine_evals += 1
                        s_db, *_ = score_config(trial_t, trial_rs)
                        return s_db

                    result = _scipy_optimize.minimize(
                        objective,
                        x0,
                        method="Nelder-Mead",
                        bounds=bounds_arr,
                        options={
                            "xatol": 1e-4,
                            "fatol": 1e-3,
                            "maxiter": 50 * max(1, len(search_dims)) * 4,
                            "adaptive": True,
                            "disp": False,
                        },
                    )

                    final_t, final_rs = unpack(np.asarray(result.x, dtype=float), cand)
                    refine_evals += 1
                    final_score, final_nom, final_worst, final_avg, final_best = score_config(
                        final_t, final_rs
                    )

                    if final_score > cand.score_db + 1e-9:
                        refined.append(cand)
                        continue

                    refined.append(
                        InverseCandidate(
                            score_db=final_score,
                            nominal_mean_db=final_nom,
                            worst_mean_db=final_worst,
                            avg_mean_db=final_avg,
                            best_mean_db=final_best,
                            thickness_in=final_t[:],
                            material_files=cand.material_files[:],
                            sheet_resistance_ohm=final_rs[:],
                        )
                    )

                refined.sort(key=lambda c: c.score_db, reverse=False)
                top_candidates = refined

            inverse_samples: list[list[list[float]]] = []
            for cand in top_candidates:
                cand_layers = build_loaded_layers(cand.thickness_in, cand.sheet_resistance_ohm)
                freq_samples = [[] for _ in target_freqs]
                for t_scale, e_scale, m_scale in scales:
                    for angle_deg in target_angles:
                        metrics = compute_angle_metrics_many(
                            target_freqs,
                            angle_deg,
                            cand_layers,
                            wave_pol,
                            thickness_scale=t_scale,
                            eps_scale=e_scale,
                            mu_scale=m_scale,
                            prepared_wave_terms=prepared_inverse_wave_terms.get(
                                (angle_deg, e_scale, m_scale)
                            ),
                        )
                        for fi, val in enumerate(metrics["metal_loss_db"]):
                            freq_samples[fi].append(val)
                inverse_samples.append(freq_samples)

            best = top_candidates[0]
            unc_state = "enabled" if uncertainty_cfg.enabled else "disabled"
            n_rs_dims = sum(1 for d in search_dims if d[1] == "rs")
            n_t_dims = len(search_dims) - n_rs_dims
            search_mode_text = (
                f"Monte Carlo, {eval_count} random sample(s) over "
                f"{n_t_dims} thickness + {n_rs_dims} sheet-R variable(s)"
            )
            seed_text_msg = (
                f", seed={search_seed}" if search_seed is not None else ""
            )
            refine_text = (
                f"Nelder-Mead, {refine_evals} extra evals"
                if refine_top_candidates
                else "disabled"
            )
            msg = (
                f"Inverse design complete.\n"
                f"Objective: {score_mode}\n"
                f"Region: {target_freq_desc}, {a_start:g}-{a_stop:g} deg, pol={wave_pol.upper()}\n"
                f"Uncertainty: {unc_state} ({len(scales)} corner(s))\n"
                f"Search: {search_mode_text}{seed_text_msg}\n"
                f"Evaluated: {eval_count} candidates (budget {max_evals})\n"
                f"Refinement: {refine_text}\n"
                f"Best score: {best.score_db:.3f} dB | nominal {best.nominal_mean_db:.3f} dB | "
                f"worst-case {best.worst_mean_db:.3f} dB\n"
                f"Stored top {len(top_candidates)} candidates. Use Apply Selected to update the stack."
            )
            return top_candidates, msg, [float(v) for v in target_freqs], inverse_samples

        def on_success(result: tuple[list[InverseCandidate], str, list[float], list[list[list[float]]]]) -> None:
            self.inverse_candidates, msg, freqs_plot, samples_plot = result
            self.inverse_plot_freqs = freqs_plot
            self.inverse_plot_samples = samples_plot
            self._refresh_inverse_results_list()
            if self.inv_results_list is not None:
                self.inv_results_list.clearSelection()
                self.inv_results_list.setCurrentRow(0)
            self.inv_results_frame.expand()
            self._update_plot()
            messagebox.showinfo("Inverse Design", msg)

        self._run_background_task("Inverse Design", worker, on_success, "Inverse Design Error")

    # --- Material Mix tab ---------------------------------------------------
    def _bind_mix_input_invalidation(self) -> None:
        """Prevent results from silently surviving a changed design problem."""
        string_inputs = (
            self.mix_rule_var,
            self.mix_objective_var,
            self.mix_thickness_var,
            self.mix_freq_mode_var,
            self.mix_freq_list_var,
            self.mix_target_start_var,
            self.mix_target_stop_var,
            self.mix_target_step_var,
            self.mix_prop_source_var,
            self.mix_prop_eps_re_var,
            self.mix_prop_eps_im_var,
            self.mix_prop_mu_re_var,
            self.mix_prop_mu_im_var,
            self.mix_prop_file_var,
            self.mix_prop_weps_var,
            self.mix_prop_wmu_var,
            self.mix_perf_metric_var,
            self.mix_perf_target_var,
            self.mix_perf_angle_start_var,
            self.mix_perf_angle_stop_var,
            self.mix_perf_angle_step_var,
            self.mix_perf_wave_pol_var,
            self.mix_max_evals_var,
            self.mix_top_n_var,
            self.mix_seed_var,
            self.mix_score_mode_var,
            self.mix_unc_t_pct_var,
            self.mix_unc_eps_pct_var,
            self.mix_unc_mu_pct_var,
        )
        boolean_inputs = (self.mix_refine_var, self.mix_uncertainty_var)
        for var in string_inputs:
            var.valueChanged.connect(lambda _value: self._invalidate_mix_results())
        for var in boolean_inputs:
            var.valueChanged.connect(lambda _value: self._invalidate_mix_results())

    def _sync_mix_freq_mode_state(self) -> None:
        mode = self.mix_freq_mode_var.get().strip().lower()
        band_enabled = mode.startswith("band")
        for entry in (
            self.mix_target_start_entry,
            self.mix_target_stop_entry,
            self.mix_target_step_entry,
        ):
            if entry is not None:
                entry.setEnabled(band_enabled)
        if self.mix_freq_list_entry is not None:
            self.mix_freq_list_entry.setEnabled(not band_enabled)

    def _sync_mix_uncertainty_state(self) -> None:
        enabled = self.mix_uncertainty_var.get()
        for entry in (self.mix_unc_t_entry, self.mix_unc_eps_entry, self.mix_unc_mu_entry):
            if entry is not None:
                entry.setEnabled(enabled)

    def _mix_objective_is_property(self) -> bool:
        text = self.mix_objective_var.get().strip().lower()
        return "target properties" in text or text.startswith("match properties")

    def _mix_objective_is_performance(self) -> bool:
        return "performance" in self.mix_objective_var.get().strip().lower()

    def _mix_objective_is_inverse(self) -> bool:
        return self._mix_objective_is_property() or self._mix_objective_is_performance()

    def _sync_mix_objective_state(self) -> None:
        property_mode = self._mix_objective_is_property()
        performance_mode = self._mix_objective_is_performance()
        inverse = property_mode or performance_mode
        if self.mix_prop_frame is not None:
            self.mix_prop_frame.setVisible(property_mode)
        if self.mix_perf_frame is not None:
            self.mix_perf_frame.setVisible(performance_mode)
        if self.mix_search_frame is not None:
            self.mix_search_frame.setVisible(inverse)
        if self.mix_run_btn is not None:
            self.mix_run_btn.setVisible(inverse)
        if self.mix_preview_btn is not None:
            self.mix_preview_btn.setText(
                "Preview current recipe" if inverse else "Calculate recipe"
            )
        if self.mix_workflow_help_label is not None:
            if property_mode:
                help_text = (
                    "Inverse workflow: set a target ε/μ and allowable volume-% "
                    "range for each material. FREDDY searches the bounded "
                    "volume-fraction simplex and reports the best recipes."
                )
            elif performance_mode:
                help_text = (
                    "Performance workflow: choose a reflection, absorption, or "
                    "transmission requirement over frequency and incidence angle. "
                    "FREDDY searches recipes whose worst grid point meets it."
                )
            else:
                help_text = (
                    "Forward workflow: enter relative volume amounts for the "
                    "known recipe. FREDDY normalizes them to volume percent and "
                    "predicts the effective ε/μ over the selected band."
                )
            self.mix_workflow_help_label.setText(help_text)
        self._sync_mix_prop_source_state()
        self._on_mix_performance_metric_changed()
        self._on_mix_model_changed()

    def _sync_mix_prop_source_state(self) -> None:
        prop = self._mix_objective_is_property()
        use_file = self.mix_prop_source_var.get().strip().lower().startswith("material")
        for entry in self.mix_prop_const_entries:
            entry.setEnabled(prop and not use_file)
        if self.mix_prop_file_entry is not None:
            self.mix_prop_file_entry.setEnabled(prop and use_file)
        if self.mix_prop_browse_btn is not None:
            self.mix_prop_browse_btn.setEnabled(prop and use_file)

    def _on_mix_performance_metric_changed(self) -> None:
        metric_label = self.mix_perf_metric_var.get()
        spec = MIX_PERFORMANCE_SPEC_BY_LABEL.get(metric_label)
        previous = getattr(self, "_mix_last_perf_metric", None)
        if spec is not None and previous is not None and previous != metric_label:
            # A percent target is not a sensible carry-over from a dB target (or
            # vice versa). Start a newly selected metric from its documented
            # default; the user can then edit it explicitly.
            self.mix_perf_target_var.set(str(spec["default_target"]))
        self._mix_last_perf_metric = metric_label
        if spec is None:
            text = "Select a supported performance metric."
        elif spec["direction"] == "at_most":
            text = f"Requirement: every point must be ≤ target {spec['unit']}"
        else:
            text = f"Requirement: every point must be ≥ target {spec['unit']}"
        if self.mix_perf_requirement_label is not None:
            self.mix_perf_requirement_label.setText(text)

    def _on_mix_model_changed(self) -> None:
        try:
            rule = normalize_mix_rule(self.mix_rule_var.get())
            description = MIX_RULE_DESCRIPTIONS[rule]
        except Exception as exc:
            description = str(exc)
        if self.mix_model_help_label is not None:
            self.mix_model_help_label.setText("Model assumptions: " + description)
        self._refresh_mix_components_list()
        self._update_plot()

    def _browse_mix_prop_file(self) -> None:
        p = filedialog.askopenfilename(title="Select target material file", parent=self, filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if p:
            self.mix_prop_file_var.set(p)

    def _coerce_mix_component(self, raw: dict) -> dict:
        result = {
            "file": str(raw.get("file", "")).strip(),
            "parts": float(raw.get("parts", 1.0)),
            "min": float(raw.get("min", 0.0)),
            "max": float(raw.get("max", 100.0)),
            "density": float(raw.get("density", 0.0)),
            "units": "volume_percent",
        }
        numeric = (result["parts"], result["min"], result["max"], result["density"])
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("Material recipe fields must be finite.")
        if result["parts"] < 0 or result["density"] < 0:
            raise ValueError("Recipe amount and density must be >= 0.")
        if not (0 <= result["min"] <= result["max"] <= 100):
            raise ValueError("Inverse volume bounds must satisfy 0 <= min <= max <= 100%.")
        return result

    def _refresh_mix_components_list(self) -> None:
        if self.mix_list is None:
            return
        row = self.mix_list.currentRow()
        self.mix_list.clear()
        total = sum(max(0.0, float(c.get("parts", 0.0))) for c in self.mix_components)
        try:
            rule = normalize_mix_rule(self.mix_rule_var.get())
        except Exception:
            rule = ""
        for i, c in enumerate(self.mix_components, start=1):
            parts = float(c.get("parts", 0.0))
            frac = (parts / total * 100.0) if total > 0 else 0.0
            name = Path(str(c.get("file", ""))).name or str(c.get("file", ""))
            density = float(c.get("density", 0.0))
            density_text = f" | ρ={density:g} g/cc" if density > 0 else ""
            role = "HOST | " if rule == "maxwell-garnett" and i == 1 else ""
            self.mix_list.addItem(
                f"{i}. {role}{name} | recipe {frac:.1f} vol% "
                f"(amount {parts:g}) | inverse bounds "
                f"{float(c.get('min', 0.0)):g}–{float(c.get('max', 100.0)):g} vol%"
                f"{density_text}"
            )
        if 0 <= row < len(self.mix_components):
            self.mix_list.setCurrentRow(row)

    def _mix_selected_idx(self) -> int | None:
        if self.mix_list is None:
            return None
        row = self.mix_list.currentRow()
        if row < 0 or row >= len(self.mix_components):
            return None
        return row

    def _invalidate_mix_results(self) -> None:
        self.mix_candidates = []
        self.mix_plot_data = []
        self.mix_preview = None
        self._refresh_mix_results_list()
        if self.mix_summary_label is not None:
            self.mix_summary_label.setText(
                "A material-mix input changed. Recalculate before using these results."
            )

    def _add_mix_component(self) -> None:
        dlg = MixComponentDialog(self, presets=BUILTIN_MATERIAL_PRESETS)
        dlg.exec()
        if dlg.result is not None:
            self.mix_components.append(dlg.result)
            self._invalidate_mix_results()
            self._refresh_mix_components_list()
            if self.mix_list is not None:
                self.mix_list.setCurrentRow(len(self.mix_components) - 1)

    def _edit_mix_component(self) -> None:
        idx = self._mix_selected_idx()
        if idx is None:
            messagebox.showwarning("Material Mix", "Select a component to edit.")
            return
        dlg = MixComponentDialog(
            self, initial=self.mix_components[idx], presets=BUILTIN_MATERIAL_PRESETS
        )
        dlg.exec()
        if dlg.result is not None:
            self.mix_components[idx] = dlg.result
            self._invalidate_mix_results()
            self._refresh_mix_components_list()
            if self.mix_list is not None:
                self.mix_list.setCurrentRow(idx)

    def _remove_mix_component(self) -> None:
        idx = self._mix_selected_idx()
        if idx is None:
            messagebox.showwarning("Material Mix", "Select a component to remove.")
            return
        del self.mix_components[idx]
        self._invalidate_mix_results()
        self._refresh_mix_components_list()

    def _parse_mix_freqs(self) -> tuple[list[float], str]:
        mode = self.mix_freq_mode_var.get().strip().lower()
        if mode.startswith("discrete"):
            target_freqs = self._parse_inverse_discrete_freqs(self.mix_freq_list_var.get())
            desc = "Discrete GHz: " + ", ".join(f"{v:g}" for v in target_freqs)
        else:
            f_start = float(self.mix_target_start_var.get().strip())
            f_stop = float(self.mix_target_stop_var.get().strip())
            f_step = float(self.mix_target_step_var.get().strip())
            target_freqs = make_frequency_sweep(f_start, f_stop, f_step)
            desc = f"Band GHz: {f_start:g}-{f_stop:g} (step {f_step:g})"
        return target_freqs, desc

    def _parse_mix_property_target(
        self, grid: list[float]
    ) -> tuple[list[complex], list[complex], float, float, str]:
        """Resolve the property-design target onto ``grid``. Returns
        (target_eps, target_mu, eps_weight, mu_weight, description)."""
        w_eps = float(self.mix_prop_weps_var.get().strip())
        w_mu = float(self.mix_prop_wmu_var.get().strip())
        if w_eps < 0 or w_mu < 0 or (w_eps + w_mu) <= 0:
            raise ValueError("Property weights must be >= 0 and not both zero.")
        source = self.mix_prop_source_var.get().strip().lower()
        if source.startswith("material"):
            path = self.mix_prop_file_var.get().strip()
            if not path:
                raise ValueError(
                    "Select a target material CSV "
                    "(5 columns: frequency_hz,eps_real,eps_imag,mu_real,mu_imag)."
                )
            table = read_material_table(Path(path), 0)
            validate_sweep_coverage(grid, table, "target material")
            target_eps = interp_complex_many(grid, table.freq_ghz, table.eps_r)
            target_mu = interp_complex_many(grid, table.freq_ghz, table.mu_r)
            desc = f"file {Path(path).name}"
        else:
            eps_c = complex(
                float(self.mix_prop_eps_re_var.get().strip()),
                float(self.mix_prop_eps_im_var.get().strip()),
            )
            mu_c = complex(
                float(self.mix_prop_mu_re_var.get().strip()),
                float(self.mix_prop_mu_im_var.get().strip()),
            )
            if not all(
                math.isfinite(value)
                for value in (eps_c.real, eps_c.imag, mu_c.real, mu_c.imag)
            ):
                raise ValueError("Target ε and μ must contain finite values.")
            if abs(eps_c) <= 1e-12 or abs(mu_c) <= 1e-12:
                raise ValueError("Target ε and μ must be non-zero finite properties.")
            if eps_c.imag > 0 or mu_c.imag > 0:
                raise ValueError(
                    "Target ε'' and μ'' must be <= 0 (loss is negative in this convention)."
                )
            target_eps = [eps_c] * len(grid)
            target_mu = [mu_c] * len(grid)
            desc = (
                f"const ε={eps_c.real:g}{eps_c.imag:+g}j, "
                f"μ={mu_c.real:g}{mu_c.imag:+g}j"
            )
        return target_eps, target_mu, w_eps, w_mu, desc

    def _parse_mix_performance_target(self) -> dict:
        spec = MIX_PERFORMANCE_SPEC_BY_LABEL.get(self.mix_perf_metric_var.get())
        if spec is None:
            raise ValueError("Select a supported stack-performance metric.")
        target = float(self.mix_perf_target_var.get().strip())
        if not math.isfinite(target):
            raise ValueError("Performance threshold must be finite.")
        if spec["unit"] == "%" and not 0.0 <= target <= 100.0:
            raise ValueError("An absorption target must be between 0 and 100%.")
        if spec["unit"] == "dB" and target > 0.0:
            raise ValueError("Passive reflection/transmission thresholds must be <= 0 dB.")

        angle_start = validate_incidence_angle(
            float(self.mix_perf_angle_start_var.get().strip())
        )
        angle_stop = validate_incidence_angle(
            float(self.mix_perf_angle_stop_var.get().strip())
        )
        if angle_stop < angle_start:
            raise ValueError("Performance angle stop must be >= start.")
        if abs(angle_stop - angle_start) <= 1e-12:
            angles = [angle_start]
        else:
            angle_step = float(self.mix_perf_angle_step_var.get().strip())
            angles = make_sweep(angle_start, angle_stop, angle_step)
        wave_pol = normalize_wave_polarization(self.mix_perf_wave_pol_var.get())
        return {
            **spec,
            "target": target,
            "angles": angles,
            "wave_pol": wave_pol,
        }

    @staticmethod
    def _mix_performance_values(metrics: dict[str, list[float]], config: dict) -> list[float]:
        values = [float(value) for value in metrics[config["metric_key"]]]
        if config["unit"] == "%":
            return [100.0 * 10.0 ** (value / 10.0) for value in values]
        return values

    @staticmethod
    def _mix_performance_gap(values: list[float], config: dict) -> float:
        if not values:
            raise ValueError("A performance target requires at least one grid point.")
        if config["direction"] == "at_most":
            return max(values) - config["target"]
        return config["target"] - min(values)

    def _evaluate_mix_performance(
        self,
        table: MaterialTable,
        thickness_in: float,
        config: dict,
        *,
        thickness_scale: float = 1.0,
        eps_scale: float = 1.0,
        mu_scale: float = 1.0,
    ) -> dict:
        layer = LoadedLayer(
            thickness_m=thickness_in * INCH_TO_M,
            anisotropic=False,
            polarization_deg=0.0,
            table_0deg=table,
            table_90deg=None,
        )
        freqs = list(table.freq_ghz)
        prepared_properties = (
            prepare_layer_properties_many(freqs, [layer]) if NUMPY_AVAILABLE else None
        )
        grid = [[0.0 for _angle in config["angles"]] for _freq in freqs]
        all_values: list[float] = []
        for angle_index, angle_deg in enumerate(config["angles"]):
            metrics = compute_angle_metrics_many(
                freqs,
                angle_deg,
                [layer],
                config["wave_pol"],
                thickness_scale=thickness_scale,
                eps_scale=eps_scale,
                mu_scale=mu_scale,
                prepared_properties=prepared_properties,
            )
            values = self._mix_performance_values(metrics, config)
            all_values.extend(values)
            for freq_index, value in enumerate(values):
                grid[freq_index][angle_index] = value
        return {
            **config,
            "freqs": freqs,
            "grid": grid,
            "gap": self._mix_performance_gap(all_values, config),
        }

    def _load_mix_components(self) -> list[dict]:
        if len(self.mix_components) < 2:
            raise ValueError("Add at least two measured materials to make a blend.")
        cache: dict[str, MaterialTable] = {}
        out: list[dict] = []
        for i, c in enumerate(self.mix_components, start=1):
            path = str(c.get("file", "")).strip()
            if not path:
                raise ValueError(f"Component {i}: property file is required.")
            key = str(Path(path))
            if key not in cache:
                cache[key] = read_material_table(Path(key), 0)
            entry = dict(c)
            entry["table"] = cache[key]
            out.append(entry)
        return out

    def _build_mix_display(
        self,
        components: list[MixComponent],
        rule: str,
        thickness_in: float,
        grid_ghz: list[float],
        target: dict | None = None,
        performance: dict | None = None,
        densities: list[float] | None = None,
        component_names: list[str] | None = None,
    ) -> dict:
        # Synthesize on the frequency grid selected in the Material Mix tab.
        # When a property target is given, also carry target curves and
        # per-frequency mismatch.
        # A model comparison at the band midpoint makes morphology sensitivity
        # visible rather than implying that one mixing law is ground truth.
        disp_table = mix_material_tables(components, rule, grid_ghz)
        lo, hi = disp_table.freq_ghz[0], disp_table.freq_ghz[-1]
        fractions = parts_to_fractions([component.parts for component in components])
        eps_tan = [
            (-value.imag / value.real) if value.real > 0 else math.nan
            for value in disp_table.eps_r
        ]
        mu_tan = [
            (-value.imag / value.real) if value.real > 0 else math.nan
            for value in disp_table.mu_r
        ]
        midpoint = disp_table.freq_ghz[len(disp_table.freq_ghz) // 2]
        comparison: list[dict] = []
        for candidate_rule in MIX_RULES:
            try:
                candidate_table = mix_material_tables(
                    components, candidate_rule, [midpoint]
                )
            except Exception:
                continue
            comparison.append(
                {
                    "rule": candidate_rule,
                    "label": MIX_RULE_LABELS[candidate_rule].split(" — ")[0],
                    "eps_re": candidate_table.eps_r[0].real,
                    "eps_im": candidate_table.eps_r[0].imag,
                    "mu_re": candidate_table.mu_r[0].real,
                    "mu_im": candidate_table.mu_r[0].imag,
                }
            )
        density_values = (
            densities
            if densities is not None and len(densities) == len(fractions)
            else None
        )
        names = (
            component_names
            if component_names is not None
            and len(component_names) == len(fractions)
            else [f"Material {index}" for index in range(1, len(fractions) + 1)]
        )
        out = {
            "freqs": list(disp_table.freq_ghz),
            "eps_re": [v.real for v in disp_table.eps_r],
            "eps_im": [v.imag for v in disp_table.eps_r],
            "mu_re": [v.real for v in disp_table.mu_r],
            "mu_im": [v.imag for v in disp_table.mu_r],
            "loss_tan_eps": eps_tan,
            "loss_tan_mu": mu_tan,
            "thickness_in": thickness_in,
            "fractions": fractions,
            "component_names": names,
            "weight_fractions": weight_fractions_from_volume(
                fractions, density_values
            ) if density_values is not None else None,
            "density_gcc": blend_density_gcc(fractions, density_values)
            if density_values is not None
            else None,
            "model": normalize_mix_rule(rule),
            "advisories": mix_model_advisories(rule, fractions),
            "comparison_frequency": midpoint,
            "model_comparison": comparison,
        }
        if target is not None:
            sel = [
                (f, te, tm)
                for f, te, tm in zip(target["freqs"], target["eps"], target["mu"])
                if lo - 1e-9 <= f <= hi + 1e-9
            ]
            if sel:
                tg_f = [s[0] for s in sel]
                tg_eps = [s[1] for s in sel]
                tg_mu = [s[2] for s in sel]
                blend_eps = interp_complex_many(tg_f, disp_table.freq_ghz, disp_table.eps_r)
                blend_mu = interp_complex_many(tg_f, disp_table.freq_ghz, disp_table.mu_r)
                out["target_freqs"] = tg_f
                out["target_eps_re"] = [v.real for v in tg_eps]
                out["target_eps_im"] = [v.imag for v in tg_eps]
                out["target_mu_re"] = [v.real for v in tg_mu]
                out["target_mu_im"] = [v.imag for v in tg_mu]
                out["err_pct"] = property_match_error_curve(
                    blend_eps, blend_mu, tg_eps, tg_mu, target["w_eps"], target["w_mu"]
                )
        if performance is not None:
            out["performance"] = self._evaluate_mix_performance(
                disp_table, thickness_in, performance
            )
        return out

    def _preview_mix(self) -> None:
        try:
            loaded = self._load_mix_components()
            rule = self.mix_rule_var.get()
            normalize_mix_rule(rule)
            thickness_in = float(self.mix_thickness_var.get().strip())
            if thickness_in <= 0:
                raise ValueError("Synthesized layer thickness must be > 0.")
            target_freqs, _desc = self._parse_mix_freqs()
            if self._mix_objective_is_property():
                t_eps, t_mu, w_eps, w_mu, _tdesc = self._parse_mix_property_target(
                    target_freqs
                )
                target = {
                    "freqs": target_freqs,
                    "eps": t_eps,
                    "mu": t_mu,
                    "w_eps": w_eps,
                    "w_mu": w_mu,
                }
            else:
                target = None
            performance = (
                self._parse_mix_performance_target()
                if self._mix_objective_is_performance()
                else None
            )
            components = [
                MixComponent(table=c["table"], parts=float(c["parts"])) for c in loaded
            ]
            if sum(c.parts for c in components) <= 0:
                raise ValueError("At least one component must have parts > 0 for a preview.")
            display = self._build_mix_display(
                components,
                rule,
                thickness_in,
                target_freqs,
                target=target,
                performance=performance,
                densities=[float(c.get("density", 0.0)) for c in loaded],
                component_names=[Path(c["file"]).name for c in loaded],
            )
        except Exception as exc:
            messagebox.showerror("Material Mix", str(exc))
            return
        self.mix_candidates = []
        self.mix_plot_data = []
        self.mix_preview = display
        self._refresh_mix_results_list()
        if self.mix_summary_label is not None:
            self.mix_summary_label.setText(self._mix_display_summary(display))
        self._update_plot()

    def _mix_display_summary(self, display: dict) -> str:
        fractions = display.get("fractions", [])
        names = display.get("component_names", [])
        recipe = " | ".join(
            f"{name}: {100 * value:.1f}%"
            for name, value in zip(names, fractions)
        )
        weight = display.get("weight_fractions")
        weight_text = (
            "\nWeight recipe: "
            + " | ".join(
                f"{name}: {100 * value:.1f}%"
                for name, value in zip(names, weight)
            )
            if weight
            else "\nWeight recipe unavailable (enter every density)"
        )
        density = display.get("density_gcc")
        density_text = f" | blend density ≈ {density:.4g} g/cc" if density else ""
        model = MIX_RULE_LABELS.get(display.get("model"), str(display.get("model", "")))
        notes = " ".join(display.get("advisories", []))
        summary = (
            f"Selected model: {model}\nVolume recipe: {recipe}{weight_text}{density_text}\n"
            f"Applicability: {notes}"
        )
        performance = display.get("performance")
        if performance is not None:
            relation = "≤" if performance["direction"] == "at_most" else "≥"
            status = "PASS" if performance["gap"] <= 0.0 else "MISS"
            summary += (
                f"\nPerformance: {performance['label']} {relation} "
                f"{performance['target']:g} {performance['unit']} | "
                f"worst requirement gap {performance['gap']:+.3f} "
                f"{performance['unit']} ({status})"
            )
        return summary

    def _run_mix_design(self) -> None:
        """Find bounded recipes for a property or stack-performance target."""
        property_mode = self._mix_objective_is_property()
        performance_mode = self._mix_objective_is_performance()
        if not (property_mode or performance_mode):
            self._preview_mix()
            return
        try:
            comp_snapshot = [
                self._coerce_mix_component(component)
                for component in self.mix_components
            ]
            if len(comp_snapshot) < 2:
                raise ValueError("Add at least two measured materials.")
            rule_norm = normalize_mix_rule(self.mix_rule_var.get())
            target_freqs, target_desc = self._parse_mix_freqs()
            target: dict | None = None
            performance_config: dict | None = None
            if property_mode:
                target_eps, target_mu, w_eps, w_mu, prop_desc = (
                    self._parse_mix_property_target(target_freqs)
                )
                target = {
                    "freqs": target_freqs,
                    "eps": target_eps,
                    "mu": target_mu,
                    "w_eps": w_eps,
                    "w_mu": w_mu,
                }
            else:
                performance_config = self._parse_mix_performance_target()
                prop_desc = ""
            thickness_in = float(self.mix_thickness_var.get().strip())
            if not math.isfinite(thickness_in) or thickness_in <= 0:
                raise ValueError("Stack-layer thickness must be finite and > 0.")
            max_evals = int(self.mix_max_evals_var.get().strip())
            top_n = int(self.mix_top_n_var.get().strip())
            if max_evals < 1 or top_n < 1:
                raise ValueError("Recipe samples and number kept must be >= 1.")
            top_n = min(top_n, max_evals)
            score_mode = self.mix_score_mode_var.get().strip()
            uncertainty_cfg = self._read_uncertainty_config(
                self.mix_uncertainty_var,
                self.mix_unc_t_pct_var,
                self.mix_unc_eps_pct_var,
                self.mix_unc_mu_pct_var,
            )
            if property_mode:
                # Effective properties do not depend on a later layer thickness.
                uncertainty_cfg = UncertaintyConfig(
                    uncertainty_cfg.enabled,
                    0.0,
                    uncertainty_cfg.eps_pct,
                    uncertainty_cfg.mu_pct,
                )
            lower = [component["min"] / 100.0 for component in comp_snapshot]
            upper = [component["max"] / 100.0 for component in comp_snapshot]
            validate_fraction_bounds(lower, upper)
            seed_text = self.mix_seed_var.get().strip()
            search_seed: int | None = int(seed_text) if seed_text else None
            refine = bool(self.mix_refine_var.get())
            if refine and (not NUMPY_AVAILABLE or not SCIPY_AVAILABLE):
                raise ValueError(
                    "Local material-recipe refinement requires NumPy and SciPy."
                )
        except Exception as exc:
            messagebox.showerror("Material Mix Error", str(exc))
            return

        def worker() -> tuple[list[MixCandidate], list[dict], str]:
            scales = build_uncertainty_scales(uncertainty_cfg)
            cache: dict[str, MaterialTable] = {}
            comps: list[dict] = []
            for index, component in enumerate(comp_snapshot, start=1):
                path = component["file"]
                if not path:
                    raise ValueError(f"Material {index}: property file is required.")
                key = str(Path(path))
                if key not in cache:
                    cache[key] = read_material_table(Path(key), 0)
                comps.append({**component, "file": key, "table": cache[key]})

            base_components = [
                MixComponent(table=component["table"], parts=1.0)
                for component in comps
            ]
            eps_cols, mu_cols = interp_components_on_grid(
                base_components, target_freqs
            )
            component_files = [component["file"] for component in comps]
            densities = [component["density"] for component in comps]

            def score_fractions(fractions: list[float]):
                table = combine_mix(
                    target_freqs, eps_cols, mu_cols, fractions, rule_norm
                )
                corner_values: list[float] = []
                nominal: float | None = None
                for t_scale, eps_scale, mu_scale in scales:
                    if property_mode:
                        error = property_match_error(
                            table.eps_r,
                            table.mu_r,
                            target["eps"],
                            target["mu"],
                            target["w_eps"],
                            target["w_mu"],
                            eps_scale,
                            mu_scale,
                        )
                    else:
                        error = self._evaluate_mix_performance(
                            table,
                            thickness_in,
                            performance_config,
                            thickness_scale=t_scale,
                            eps_scale=eps_scale,
                            mu_scale=mu_scale,
                        )["gap"]
                    corner_values.append(error)
                    if is_nominal_scale(t_scale, eps_scale, mu_scale):
                        nominal = error
                if nominal is None:
                    nominal = corner_values[0]
                worst = max(corner_values)
                average = sum(corner_values) / len(corner_values)
                score = worst if "worst" in score_mode.lower() else average
                return score, nominal, worst, average, min(corner_values)

            rng = random.Random(search_seed)
            current_amounts = [component["parts"] for component in comps]
            if sum(current_amounts) > 0:
                seed_values = parts_to_fractions(current_amounts)
            else:
                # An all-zero forward recipe is still meaningful while setting
                # inverse bounds; begin the search at the middle of those bounds.
                seed_values = [0.5 * (lo + hi) for lo, hi in zip(lower, upper)]
            recipe_seed = project_bounded_fractions(seed_values, lower, upper)
            proposals = [recipe_seed]
            seen = {tuple(round(value, 12) for value in recipe_seed)}
            attempts = 0
            while len(proposals) < max_evals and attempts < max_evals * 30:
                attempts += 1
                # Exponential variates normalized to a simplex give broad
                # coverage; projection then enforces the engineer's vol-% bounds.
                raw = [-math.log(max(rng.random(), 1e-15)) for _ in comps]
                proposal = project_bounded_fractions(
                    parts_to_fractions(raw), lower, upper
                )
                key = tuple(round(value, 10) for value in proposal)
                if key not in seen:
                    seen.add(key)
                    proposals.append(proposal)

            candidates_raw: list[dict] = []
            invalid_count = 0
            for fractions in proposals:
                try:
                    score, nominal, worst, average, best = score_fractions(fractions)
                except ValueError:
                    invalid_count += 1
                    continue
                candidates_raw.append(
                    {
                        "score": score,
                        "nominal": nominal,
                        "worst": worst,
                        "avg": average,
                        "best": best,
                        "fractions": fractions,
                    }
                )
            candidates_raw.sort(key=lambda candidate: candidate["score"])
            candidates_raw = candidates_raw[:top_n]
            if not candidates_raw:
                raise ValueError(
                    "No physically valid recipe was found for these model "
                    "assumptions and volume bounds."
                )

            refine_evals = 0
            if refine and NUMPY_AVAILABLE:
                refined: list[dict] = []
                bounds = list(zip(lower, upper))
                constraint = {
                    "type": "eq",
                    "fun": lambda x: float(np.sum(x) - 1.0),
                }
                for candidate in candidates_raw:
                    def objective(x: "np.ndarray") -> float:
                        nonlocal refine_evals
                        refine_evals += 1
                        try:
                            fractions = project_bounded_fractions(
                                [float(value) for value in x], lower, upper
                            )
                            return score_fractions(fractions)[0]
                        except ValueError:
                            return 1e12

                    result = _scipy_optimize.minimize(
                        objective,
                        np.asarray(candidate["fractions"], dtype=float),
                        method="SLSQP",
                        bounds=bounds,
                        constraints=(constraint,),
                        options={"ftol": 1e-9, "maxiter": 300, "disp": False},
                    )
                    try:
                        fractions = project_bounded_fractions(
                            [float(value) for value in result.x], lower, upper
                        )
                        score, nominal, worst, average, best = score_fractions(fractions)
                        refined_candidate = {
                            "score": score,
                            "nominal": nominal,
                            "worst": worst,
                            "avg": average,
                            "best": best,
                            "fractions": fractions,
                        }
                        refined.append(
                            refined_candidate
                            if score <= candidate["score"] + 1e-10
                            else candidate
                        )
                    except Exception:
                        refined.append(candidate)
                refined.sort(key=lambda candidate: candidate["score"])
                candidates_raw = refined[:top_n]

            candidates: list[MixCandidate] = []
            plot_data: list[dict] = []
            for candidate in candidates_raw:
                fractions = candidate["fractions"]
                candidates.append(
                    MixCandidate(
                        score_db=candidate["score"],
                        nominal_mean_db=candidate["nominal"],
                        worst_mean_db=candidate["worst"],
                        avg_mean_db=candidate["avg"],
                        best_mean_db=candidate["best"],
                        fractions=list(fractions),
                        thickness_in=thickness_in,
                        component_files=list(component_files),
                        rule=rule_norm,
                        objective_kind="property" if property_mode else "performance",
                        score_unit="%" if property_mode else performance_config["unit"],
                        weight_fractions=weight_fractions_from_volume(
                            fractions, densities
                        ),
                        density_gcc=blend_density_gcc(fractions, densities),
                    )
                )
                display_components = [
                    MixComponent(table=comps[i]["table"], parts=fractions[i])
                    for i in range(len(comps))
                ]
                plot_data.append(
                    self._build_mix_display(
                        display_components,
                        rule_norm,
                        thickness_in,
                        target_freqs,
                        target=target if property_mode else None,
                        performance=performance_config if performance_mode else None,
                        densities=densities,
                        component_names=[Path(path).name for path in component_files],
                    )
                )

            best_candidate = candidates[0]
            recipe = " | ".join(
                f"{Path(path).name}: {100 * fraction:.1f}%"
                for path, fraction in zip(
                    best_candidate.component_files, best_candidate.fractions
                )
            )
            advisories = " ".join(
                mix_model_advisories(rule_norm, best_candidate.fractions)
            )
            if property_mode:
                target_text = f"Target: {prop_desc}; {target_desc}"
                result_text = (
                    f"Match error: {best_candidate.score_db:.3f}% "
                    f"(nominal {best_candidate.nominal_mean_db:.3f}%, "
                    f"worst {best_candidate.worst_mean_db:.3f}%)"
                )
            else:
                relation = "<=" if performance_config["direction"] == "at_most" else ">="
                angles = performance_config["angles"]
                status = "PASS" if best_candidate.worst_mean_db <= 0.0 else "MISS"
                target_text = (
                    f"Target: {performance_config['label']} {relation} "
                    f"{performance_config['target']:g} {performance_config['unit']}; "
                    f"{target_desc}; angles {angles[0]:g}-{angles[-1]:g} deg; "
                    f"pol={performance_config['wave_pol'].upper()}"
                )
                result_text = (
                    f"Search score: {best_candidate.score_db:+.3f} "
                    f"{best_candidate.score_unit}; certified worst-corner gap "
                    f"{best_candidate.worst_mean_db:+.3f} {best_candidate.score_unit} "
                    f"({status}; gap <= 0 passes); nominal gap "
                    f"{best_candidate.nominal_mean_db:+.3f}"
                )
            message = (
                "Inverse material recipe search complete.\n"
                f"{target_text}\n"
                f"Model: {MIX_RULE_LABELS[rule_norm]}\n"
                f"Best volume recipe: {recipe}\n"
                f"{result_text}\n"
                f"Evaluated {len(proposals)} bounded recipe(s); "
                f"{invalid_count} invalid under model; refinement evaluations "
                f"{refine_evals}.\nApplicability: {advisories}"
            )
            return candidates, plot_data, message

        def on_success(result: tuple[list[MixCandidate], list[dict], str]) -> None:
            self.mix_candidates, self.mix_plot_data, message = result
            self.mix_preview = None
            self._refresh_mix_results_list()
            if self.mix_results_list is not None:
                self.mix_results_list.setCurrentRow(0)
            if self.mix_results_frame is not None:
                self.mix_results_frame.expand()
            if self.mix_summary_label is not None and self.mix_plot_data:
                candidate = self.mix_candidates[0]
                if candidate.objective_kind == "performance":
                    status = "PASS" if candidate.worst_mean_db <= 0.0 else "MISS"
                    score_text = (
                        f"\nWorst uncertainty-corner gap: "
                        f"{candidate.worst_mean_db:+.3f} {candidate.score_unit} "
                        f"({status}; gap <= 0 passes)"
                    )
                else:
                    score_text = f"\nBest target mismatch: {candidate.score_db:.3f}%"
                self.mix_summary_label.setText(
                    self._mix_display_summary(self.mix_plot_data[0]) + score_text
                )
            self._update_plot()
            messagebox.showinfo("Material Mix", message)

        self._run_background_task(
            "Material Mix", worker, on_success, "Material Mix Error"
        )

    def _refresh_mix_results_list(self) -> None:
        if self.mix_results_list is None:
            return
        self.mix_results_list.clear()
        for i, c in enumerate(self.mix_candidates, start=1):
            frac_text = " | ".join(
                f"{Path(path).stem}: {fraction * 100:.0f}%"
                for path, fraction in zip(c.component_files, c.fractions)
            )
            wt_text = (
                " | wt=["
                + " | ".join(
                    f"{Path(path).stem}: {fraction * 100:.0f}%"
                    for path, fraction in zip(c.component_files, c.weight_fractions)
                )
                + "]"
                if c.weight_fractions
                else ""
            )
            if c.objective_kind == "performance":
                status = "PASS" if c.worst_mean_db <= 0.0 else "MISS"
                score_text = (
                    f"score={c.score_db:+.2f} {c.score_unit} | "
                    f"nom gap={c.nominal_mean_db:+.2f} | "
                    f"worst gap={c.worst_mean_db:+.2f} {status}"
                )
            else:
                score_text = (
                    f"err={c.score_db:.2f}% | nom={c.nominal_mean_db:.2f}% | "
                    f"worst={c.worst_mean_db:.2f}%"
                )
            self.mix_results_list.addItem(
                f"{i:02d}: {score_text} | vol=[{frac_text}]{wt_text}"
            )

    def _current_mix_material(self) -> tuple[MaterialTable, float, str]:
        if self.mix_candidates:
            idx = 0
            if self.mix_results_list is not None and self.mix_results_list.currentRow() >= 0:
                idx = self.mix_results_list.currentRow()
            idx = max(0, min(idx, len(self.mix_candidates) - 1))
            cand = self.mix_candidates[idx]
            loaded = self._load_mix_components()
            if len(loaded) != len(cand.fractions):
                raise ValueError(
                    "Components changed since the search ran. Re-run the search before applying."
                )
            components = [
                MixComponent(table=loaded[i]["table"], parts=cand.fractions[i])
                for i in range(len(loaded))
            ]
            if idx >= len(self.mix_plot_data):
                raise ValueError("Candidate frequency grid is unavailable; re-run the search.")
            grid = [float(value) for value in self.mix_plot_data[idx]["freqs"]]
            table = mix_material_tables(components, cand.rule, grid)
            return table, cand.thickness_in, f"blend candidate #{idx + 1}"
        if self.mix_preview is not None:
            loaded = self._load_mix_components()
            components = [
                MixComponent(table=c["table"], parts=float(c["parts"])) for c in loaded
            ]
            grid = [float(value) for value in self.mix_preview["freqs"]]
            table = mix_material_tables(
                components, self.mix_rule_var.get(), grid
            )
            thickness_in = float(self.mix_thickness_var.get().strip())
            return table, thickness_in, "previewed blend"
        raise ValueError("Preview a blend or run a search first.")

    def _export_mix_material(self) -> None:
        try:
            table, _thickness, label = self._current_mix_material()
        except Exception as exc:
            messagebox.showerror("Material Mix", str(exc))
            return
        path_str = filedialog.asksaveasfilename(
            title="Export mixed material",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )
        if not path_str:
            return
        try:
            write_material_table(Path(path_str), table)
            self._publish_nominal_artifact("material", Path(path_str))
        except Exception as exc:
            messagebox.showerror("Material Mix", str(exc))
            return
        messagebox.showinfo("Material Mix", f"Exported {label} to:\n{path_str}")

    def _apply_mix_as_layer(self) -> None:
        try:
            table, thickness_in, label = self._current_mix_material()
        except Exception as exc:
            messagebox.showerror("Material Mix", str(exc))
            return
        path_str = filedialog.asksaveasfilename(
            title="Save mixed material for the new layer",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )
        if not path_str:
            return
        try:
            write_material_table(Path(path_str), table)
            self._publish_nominal_artifact("material", Path(path_str))
            if thickness_in <= 0:
                thickness_in = 0.125
            self.layers.append(
                LayerConfig(
                    thickness_in=thickness_in,
                    anisotropic=False,
                    file_0deg=path_str,
                    file_90deg="",
                    polarization_deg=0.0,
                )
            )
            self._refresh_layers()
        except Exception as exc:
            messagebox.showerror("Material Mix", str(exc))
            return
        messagebox.showinfo(
            "Material Mix",
            f"Saved {label} and added it as a new layer:\n{path_str}",
        )

    def _draw_mix_placeholder(self, text: str) -> None:
        if not MPL_AVAILABLE or self.ax_heatmap is None or self.canvas is None:
            return
        if self.heatmap_cbar is not None:
            self.heatmap_cbar.remove()
            self.heatmap_cbar = None
        colors = self._colors
        self.ax_heatmap.clear()
        self.ax_heatmap.set_title("Synthesized Material")
        self.ax_heatmap.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            transform=self.ax_heatmap.transAxes,
            color=colors["muted_text"],
        )
        self._style_plot_axis(self.ax_heatmap)
        self.ax_heatmap.grid(False)
        for ax, title in (
            (self.ax_freq_slice, "Loss tangent"),
            (self.ax_angle_slice, "Model sensitivity"),
        ):
            if ax is not None:
                ax.clear()
                ax.set_title(title, fontsize=9, pad=2)
                self._style_plot_axis(ax)
                ax.grid(False)
        self.canvas.draw_idle()

    @staticmethod
    def _mix_plot_edges(values: list[float]) -> list[float]:
        """Convert monotonic sample centers to plotting-cell edges."""
        if len(values) == 1:
            span = max(0.5, abs(values[0]) * 0.02)
            return [values[0] - span, values[0] + span]
        mids = [0.5 * (a + b) for a, b in zip(values[:-1], values[1:])]
        return [values[0] - (mids[0] - values[0]), *mids, values[-1] + (values[-1] - mids[-1])]

    def _draw_mix_performance_plot(
        self, data: dict, performance: dict, subtitle: str, selected_idx: int
    ) -> None:
        """Draw the selected blend's full frequency/angle requirement grid."""
        colors = self._colors
        freqs = [float(value) for value in performance["freqs"]]
        angles = [float(value) for value in performance["angles"]]
        grid = performance["grid"]
        direction = performance["direction"]
        target = float(performance["target"])
        unit = performance["unit"]
        relation = "≤" if direction == "at_most" else "≥"

        self.ax_heatmap.clear()
        image = self.ax_heatmap.pcolormesh(
            self._mix_plot_edges(angles),
            self._mix_plot_edges(freqs),
            grid,
            shading="flat",
            cmap="viridis",
        )
        flat_grid = [value for row in grid for value in row]
        if (
            len(freqs) >= 2
            and len(angles) >= 2
            and min(flat_grid) <= target <= max(flat_grid)
        ):
            contour = self.ax_heatmap.contour(
                angles,
                freqs,
                grid,
                levels=[target],
                colors=[colors["plot_worst"]],
                linewidths=1.4,
            )
            if contour.allsegs and any(len(segment) > 0 for segment in contour.allsegs[0]):
                self.ax_heatmap.clabel(contour, fmt={target: f"target {target:g}"}, fontsize=7)
        self.ax_heatmap.set_title(f"{performance['label']} ({subtitle})")
        self.ax_heatmap.set_xlabel("Incidence angle (deg)")
        self.ax_heatmap.set_ylabel("Frequency (GHz)")
        self._style_plot_axis(self.ax_heatmap)
        self.ax_heatmap.grid(False)
        self.heatmap_cbar = self.fig.colorbar(image, ax=self.ax_heatmap)
        self.heatmap_cbar.set_label(f"{performance['label']} [{unit}]")
        style_colorbar(self.heatmap_cbar, colors)

        reducer = max if direction == "at_most" else min
        worst_by_freq = [reducer(row) for row in grid]
        worst_by_angle = [
            reducer(grid[freq_index][angle_index] for freq_index in range(len(freqs)))
            for angle_index in range(len(angles))
        ]

        self.ax_freq_slice.clear()
        self.ax_freq_slice.plot(
            freqs,
            worst_by_freq,
            color=colors["plot_line_freq"],
            linewidth=1.8,
            label="worst angle",
        )
        self.ax_freq_slice.axhline(
            target,
            color=colors["plot_worst"],
            linewidth=1.2,
            linestyle="--",
            label=f"target {relation} {target:g}",
        )
        self.ax_freq_slice.set_title("Worst angle at each frequency", fontsize=9, pad=2)
        self.ax_freq_slice.set_xlabel("Frequency (GHz)", fontsize=8)
        self.ax_freq_slice.set_ylabel(unit, fontsize=8)
        self._style_plot_axis(self.ax_freq_slice)
        self.ax_freq_slice.grid(True, color=colors["plot_grid"], alpha=0.3)
        self.ax_freq_slice.legend(loc="best", fontsize=7)

        self.ax_angle_slice.clear()
        self.ax_angle_slice.plot(
            angles,
            worst_by_angle,
            color=colors["plot_line_angle"],
            linewidth=1.8,
            label="worst frequency",
        )
        self.ax_angle_slice.axhline(
            target,
            color=colors["plot_worst"],
            linewidth=1.2,
            linestyle="--",
            label=f"target {relation} {target:g}",
        )
        self.ax_angle_slice.set_title("Worst frequency at each angle", fontsize=9, pad=2)
        self.ax_angle_slice.set_xlabel("Incidence angle (deg)", fontsize=8)
        self.ax_angle_slice.set_ylabel(unit, fontsize=8)
        self._style_plot_axis(self.ax_angle_slice)
        self.ax_angle_slice.grid(True, color=colors["plot_grid"], alpha=0.3)
        self.ax_angle_slice.legend(loc="best", fontsize=7)

        if self.mix_summary_label is not None:
            summary = self._mix_display_summary(data)
            if self.mix_candidates and 0 <= selected_idx < len(self.mix_candidates):
                candidate = self.mix_candidates[selected_idx]
                status = "PASS" if candidate.worst_mean_db <= 0.0 else "MISS"
                summary += (
                    f"\nSearch score: {candidate.score_db:+.3f} "
                    f"{candidate.score_unit}; worst gap "
                    f"{candidate.worst_mean_db:+.3f} {candidate.score_unit} "
                    f"({status}; gap ≤ 0 passes)"
                )
            self.mix_summary_label.setText(summary)
        self.canvas.draw_idle()

    def _update_mix_plot(self) -> None:
        if (
            not MPL_AVAILABLE
            or self.ax_heatmap is None
            or self.ax_freq_slice is None
            or self.ax_angle_slice is None
            or self.canvas is None
        ):
            return
        if self.heatmap_cbar is not None:
            self.heatmap_cbar.remove()
            self.heatmap_cbar = None

        data: dict | None = None
        subtitle = ""
        selected_idx = 0
        if self.mix_candidates and self.mix_plot_data:
            if self.mix_results_list is not None and self.mix_results_list.currentRow() >= 0:
                selected_idx = self.mix_results_list.currentRow()
            selected_idx = max(0, min(selected_idx, len(self.mix_plot_data) - 1))
            data = self.mix_plot_data[selected_idx]
            subtitle = f"candidate #{selected_idx + 1}"
        elif self.mix_preview is not None:
            data = self.mix_preview
            subtitle = "preview"

        if data is None:
            self._draw_mix_placeholder(
                "Preview a blend or run a search to view synthesized ε/μ."
            )
            return

        colors = self._colors
        freqs = data["freqs"]

        performance = data.get("performance")
        if performance is not None:
            self._draw_mix_performance_plot(
                data, performance, subtitle, selected_idx
            )
            return

        has_target = "target_freqs" in data

        self.ax_heatmap.clear()
        self.ax_heatmap.plot(freqs, data["eps_re"], color=colors["plot_line_freq"], linewidth=1.8, label="eps'")
        self.ax_heatmap.plot(freqs, data["eps_im"], color=colors["plot_worst"], linewidth=1.4, linestyle="--", label="eps''")
        self.ax_heatmap.plot(freqs, data["mu_re"], color=colors["plot_line_angle"], linewidth=1.8, label="mu'")
        self.ax_heatmap.plot(freqs, data["mu_im"], color=colors["plot_crosshair"], linewidth=1.4, linestyle="--", label="mu''")
        if has_target:
            tf = data["target_freqs"]
            for key, color, label in (
                ("target_eps_re", "plot_line_freq", "target eps'"),
                ("target_eps_im", "plot_worst", "target eps''"),
                ("target_mu_re", "plot_line_angle", "target mu'"),
                ("target_mu_im", "plot_crosshair", "target mu''"),
            ):
                self.ax_heatmap.plot(
                    tf,
                    data[key],
                    color=colors[color],
                    linewidth=1.2,
                    linestyle=":",
                    alpha=0.75,
                    label=label,
                )
        self.ax_heatmap.set_title(
            f"Synthesized ε/μ vs target ({subtitle})" if has_target
            else f"Synthesized ε/μ ({subtitle})"
        )
        self.ax_heatmap.set_xlabel("Frequency (GHz)")
        self.ax_heatmap.set_ylabel("Relative ε, μ")
        self._style_plot_axis(self.ax_heatmap)
        self.ax_heatmap.grid(True, color=colors["plot_grid"], alpha=0.3)
        self.ax_heatmap.legend(loc="best", fontsize=7, ncol=2 if has_target else 1)

        self.ax_freq_slice.clear()
        if has_target and data.get("err_pct"):
            self.ax_freq_slice.plot(
                data["target_freqs"],
                data["err_pct"],
                color=colors["plot_line_angle"],
                linewidth=1.8,
            )
            self.ax_freq_slice.set_title("Property match error", fontsize=9, pad=2)
            self.ax_freq_slice.set_ylabel("Mismatch (%)", fontsize=8)
        else:
            self.ax_freq_slice.plot(
                freqs,
                data["loss_tan_eps"],
                color=colors["plot_line_freq"],
                linewidth=1.8,
                label="tan δε",
            )
            self.ax_freq_slice.plot(
                freqs,
                data["loss_tan_mu"],
                color=colors["plot_line_angle"],
                linewidth=1.8,
                label="tan δμ",
            )
            self.ax_freq_slice.set_title("Effective loss tangent", fontsize=9, pad=2)
            self.ax_freq_slice.set_ylabel("Loss tangent", fontsize=8)
            self.ax_freq_slice.legend(loc="best", fontsize=7)
        self.ax_freq_slice.set_xlabel("Frequency (GHz)", fontsize=8)
        self._style_plot_axis(self.ax_freq_slice)
        self.ax_freq_slice.grid(True, color=colors["plot_grid"], alpha=0.3)

        self.ax_angle_slice.clear()
        if self.mix_candidates:
            score_unit = "%"
            ranks = list(range(1, len(self.mix_candidates) + 1))
            scores = [c.score_db for c in self.mix_candidates]
            self.ax_angle_slice.bar(ranks, scores, color=colors["plot_line_freq"], alpha=0.7)
            if 0 <= selected_idx < len(ranks):
                self.ax_angle_slice.bar(
                    [ranks[selected_idx]],
                    [scores[selected_idx]],
                    color=colors["plot_line_angle"],
                    zorder=3,
                )
            self.ax_angle_slice.set_title("Blend scores by rank", fontsize=9, pad=2)
            self.ax_angle_slice.set_xlabel("Candidate rank", fontsize=8)
            self.ax_angle_slice.set_ylabel(f"Score ({score_unit})", fontsize=8)
        else:
            comparisons = data.get("model_comparison", [])
            if comparisons:
                positions = list(range(len(comparisons)))
                width = 0.38
                self.ax_angle_slice.bar(
                    [position - width / 2 for position in positions],
                    [entry["eps_re"] for entry in comparisons],
                    width=width,
                    color=colors["plot_line_freq"],
                    alpha=0.75,
                    label="ε′",
                )
                self.ax_angle_slice.bar(
                    [position + width / 2 for position in positions],
                    [entry["mu_re"] for entry in comparisons],
                    width=width,
                    color=colors["plot_line_angle"],
                    alpha=0.75,
                    label="μ′",
                )
                self.ax_angle_slice.set_xticks(positions)
                self.ax_angle_slice.set_xticklabels(
                    [entry["label"] for entry in comparisons],
                    rotation=28,
                    ha="right",
                    fontsize=6,
                )
                self.ax_angle_slice.set_title(
                    f"Model sensitivity @ {data['comparison_frequency']:g} GHz",
                    fontsize=9,
                    pad=2,
                )
                self.ax_angle_slice.set_ylabel("Real effective property", fontsize=8)
                self.ax_angle_slice.legend(loc="best", fontsize=7)
            else:
                self.ax_angle_slice.set_title(
                    "No alternate models applicable", fontsize=9, pad=2
                )
        self._style_plot_axis(self.ax_angle_slice)
        self.ax_angle_slice.grid(True, color=colors["plot_grid"], alpha=0.3)

        if self.mix_summary_label is not None:
            summary = self._mix_display_summary(data)
            if self.mix_candidates and 0 <= selected_idx < len(self.mix_candidates):
                candidate = self.mix_candidates[selected_idx]
                summary += f"\nTarget mismatch: {candidate.score_db:.3f}%"
            self.mix_summary_label.setText(summary)

        self.canvas.draw_idle()

    def _export_ibc_batch(self) -> None:
        try:
            if not self.layers:
                raise ValueError("Add at least one layer.")
            layer_index = self._selected_ibc_batch_layer_index()
            plan = self._plan_ibc_batch()
            layer_snapshot = self._snapshot_layers()
            # The selected layer is overwritten for every output. Seed only
            # the frozen worker snapshot so loading does not depend on its
            # current nominal thickness, while the live stack stays unchanged.
            layer_snapshot[layer_index].thickness_in = plan[0].thickness_in
            frequencies = make_frequency_sweep(
                float(self.f_start_var.get().strip()),
                float(self.f_stop_var.get().strip()),
                float(self.f_step_var.get().strip()),
            )
        except Exception as exc:
            messagebox.showerror("IBC Batch", str(exc), parent=self)
            return

        if not self._confirm_output_replacements(
            [item.path for item in plan], operation="IBC Batch"
        ):
            return

        def worker() -> dict[str, object]:
            loaded_layers = self._load_layers(0, layer_snapshot)
            count = export_pec_ibc_thickness_batch(
                plan, loaded_layers, layer_index, frequencies
            )
            return {"count": count, "frequency_count": len(frequencies)}

        def on_success(result: dict[str, object]) -> None:
            # A multi-file batch has no single honest "current" artifact for
            # GHOST. Only a one-file batch is unambiguous enough to publish via
            # the host's singular nominal-artifact signal.
            if len(plan) == 1:
                try:
                    self._publish_nominal_artifact("ibc", plan[0].path)
                except Exception as exc:
                    messagebox.showerror(
                        "Attachable IBC Export",
                        "The IBC CSV was written but could not be made available "
                        f"to GHOST:\n{exc}",
                        parent=self,
                    )
            folder = plan[0].path.parent.resolve()
            first = plan[0].path.name
            last = plan[-1].path.name
            detail = first if len(plan) == 1 else f"{first}\nthrough\n{last}"
            message = (
                f"Wrote {int(result['count'])} nominal PEC-backed IBC CSV(s), "
                f"each with {int(result['frequency_count'])} frequency points, to:\n"
                f"{folder}\n\n{detail}"
            )
            if len(plan) > 1:
                message += (
                    "\n\nNo one batch file was auto-selected for GHOST; choose "
                    "the thickness-specific CSV you want to attach."
                )
            messagebox.showinfo("IBC Batch Complete", message, parent=self)

        self._run_background_task(
            "IBC Batch", worker, on_success, "IBC Batch Error"
        )

    def _compute_impedance(self) -> None:
        try:
            if not self.layers:
                raise ValueError("Add at least one layer.")

            layer_snapshot = self._snapshot_layers()
            output_path = Path(self.output_var.get().strip())
            uncertainty = self._read_uncertainty_config(
                self.uncertainty_var,
                self.unc_t_pct_var,
                self.unc_eps_pct_var,
                self.unc_mu_pct_var,
            )
            uncertainty_has_bounds = uncertainty.enabled and any(
                value > 0
                for value in (
                    uncertainty.thickness_pct,
                    uncertainty.eps_pct,
                    uncertainty.mu_pct,
                )
            )
            f_start = float(self.f_start_var.get().strip())
            f_stop = float(self.f_stop_var.get().strip())
            f_step = float(self.f_step_var.get().strip())
            freqs = make_frequency_sweep(f_start, f_stop, f_step)
            backing = normalize_backing(self.backing_var.get())
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        planned_outputs = [output_path]
        if uncertainty_has_bounds:
            planned_outputs.append(uncertainty_report_path(output_path))
        if not self._confirm_output_replacements(
            planned_outputs, operation="Impedance"
        ):
            return

        # Impedance is a broadside (normal-incidence) solve; polarization is unused.
        wave_pol = normalize_wave_polarization("TE")

        def worker() -> dict[str, object]:
            loaded_layers = self._load_layers(0, layer_snapshot)
            n, summary = self._compute_frequency_mode(
                output_path,
                True,
                loaded_layers,
                backing,
                uncertainty,
                freqs,
                wave_pol,
            )
            return {"count": n, "summary": summary}

        def on_success(result: dict[str, object]) -> None:
            self.last_heatmap_results = None
            self.last_heatmap_uncertainty_min = None
            self.last_heatmap_uncertainty_max = None
            self.selected_x_idx = None
            self.selected_freq_idx = None
            self._update_plot()
            # Only the PEC-backed broadside result is a physically suitable
            # one-sided IBC for a closed Type 2 GHOST body. Air-backed and all
            # other analysis products intentionally never enter the handoff.
            if backing == "pec":
                try:
                    self._publish_nominal_artifact("ibc", output_path)
                except Exception as exc:
                    messagebox.showerror(
                        "Attachable IBC Export",
                        "The nominal CSV was written but could not be made "
                        f"available to GHOST:\n{exc}",
                    )
            message = (
                f"Wrote {int(result['count'])} nominal solver-compatible "
                f"frequency points to:\n{output_path}"
            )
            if uncertainty_has_bounds:
                message += (
                    "\n\nUncertainty bounds were written separately to:\n"
                    f"{uncertainty_report_path(output_path)}"
                )
            if backing == "air":
                messagebox.showwarning(
                    "Complete — air-backed analysis caution",
                    message
                    + "\n\nThis air-terminated input impedance is a planar "
                    "analysis result. It is not generally a physically "
                    "equivalent one-sided IBC for a closed transmitting RCS "
                    "body. Use PEC backing for a coating collapsed onto a "
                    "Type 2 conductor, or export/model dielectric layers "
                    "explicitly.",
                )
            else:
                messagebox.showinfo("Complete", message)

        self._run_background_task("Impedance", worker, on_success, "Error")

    def _compute_off_angle(self) -> None:
        try:
            if not self.layers:
                raise ValueError("Add at least one layer.")

            layer_snapshot = self._snapshot_layers()
            output_path = Path(self.angle_output_var.get().strip())
            uncertainty = self._read_uncertainty_config(
                self.angle_uncertainty_var,
                self.angle_unc_t_pct_var,
                self.angle_unc_eps_pct_var,
                self.angle_unc_mu_pct_var,
            )
            f_start = float(self.angle_f_start_var.get().strip())
            f_stop = float(self.angle_f_stop_var.get().strip())
            f_step = float(self.angle_f_step_var.get().strip())
            freqs = make_frequency_sweep(f_start, f_stop, f_step)
            wave_pol = normalize_wave_polarization(self.wave_pol_var.get())

            a_start = float(self.angle_start_var.get().strip())
            a_stop = float(self.angle_stop_var.get().strip())
            a_start = validate_incidence_angle(a_start)
            a_stop = validate_incidence_angle(a_stop)
            if a_stop < a_start:
                raise ValueError("Angle stop must be >= start.")
            if abs(a_stop - a_start) <= 1e-12:
                angles = [a_start]
            else:
                a_step = float(self.angle_step_var.get().strip())
                angles = make_sweep(a_start, a_stop, a_step)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        if not self._confirm_output_replacements(
            [output_path], operation="Off-Angle"
        ):
            return

        def worker() -> dict[str, object]:
            loaded_layers = self._load_layers(0, layer_snapshot)
            n, out, env_min, env_max, summary = self._compute_angle_mode(
                output_path,
                True,
                loaded_layers,
                uncertainty,
                angles,
                freqs,
                wave_pol,
            )
            return {
                "count": n,
                "summary": summary,
                "out": out,
                "env_min": env_min,
                "env_max": env_max,
            }

        def on_success(result: dict[str, object]) -> None:
            self.last_heatmap_results = result["out"]  # type: ignore[assignment]
            self.last_heatmap_uncertainty_min = result["env_min"]  # type: ignore[assignment]
            self.last_heatmap_uncertainty_max = result["env_max"]  # type: ignore[assignment]
            self.selected_x_idx = None
            self.selected_freq_idx = None
            self._update_plot()
            messagebox.showinfo("Complete", f"Wrote {int(result['count'])} heatmap points to:\n{output_path}")

        self._run_background_task("Off Angle", worker, on_success, "Error")

    def _compute_thickness(self) -> None:
        try:
            if not self.layers:
                raise ValueError("Add at least one layer.")

            layer_idx = self._selected_thickness_layer_index()
            layer_snapshot = self._snapshot_layers()
            output_path = Path(self.thk_output_var.get().strip())
            uncertainty = self._read_uncertainty_config(
                self.thk_uncertainty_var,
                self.thk_unc_t_pct_var,
                self.thk_unc_eps_pct_var,
                self.thk_unc_mu_pct_var,
            )
            f_start = float(self.thk_f_start_var.get().strip())
            f_stop = float(self.thk_f_stop_var.get().strip())
            f_step = float(self.thk_f_step_var.get().strip())
            freqs = make_frequency_sweep(f_start, f_stop, f_step)
            wave_pol = normalize_wave_polarization(self.thk_wave_pol_var.get())

            angle_deg = float(self.thk_angle_var.get().strip())
            angle_deg = validate_incidence_angle(angle_deg)

            t_start = float(self.thk_start_var.get().strip())
            t_stop = float(self.thk_stop_var.get().strip())
            if t_start <= 0:
                raise ValueError("Thickness start must be > 0 in.")
            if t_stop < t_start:
                raise ValueError("Thickness stop must be >= start.")
            if abs(t_stop - t_start) <= 1e-12:
                thicknesses = [t_start]
            else:
                t_step = float(self.thk_step_var.get().strip())
                thicknesses = make_sweep(t_start, t_stop, t_step)
            # The swept layer's own thickness is overwritten per column, so a
            # zero-thickness placeholder in the stack is not an error here.
            layer_snapshot[layer_idx].thickness_in = t_start
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            return

        if not self._confirm_output_replacements(
            [output_path], operation="Thickness"
        ):
            return

        def worker() -> dict[str, object]:
            loaded_layers = self._load_layers(0, layer_snapshot)
            n, out, env_min, env_max, summary = self._compute_thickness_mode(
                output_path,
                True,
                loaded_layers,
                layer_idx,
                uncertainty,
                thicknesses,
                freqs,
                wave_pol,
                angle_deg,
            )
            return {
                "count": n,
                "summary": summary,
                "out": out,
                "env_min": env_min,
                "env_max": env_max,
            }

        def on_success(result: dict[str, object]) -> None:
            self.last_thickness_results = result["out"]  # type: ignore[assignment]
            self.last_thickness_uncertainty_min = result["env_min"]  # type: ignore[assignment]
            self.last_thickness_uncertainty_max = result["env_max"]  # type: ignore[assignment]
            self.selected_x_idx = None
            self.selected_freq_idx = None
            self._update_plot()
            messagebox.showinfo(
                "Complete", f"Wrote {int(result['count'])} heatmap points to:\n{output_path}"
            )

        self._run_background_task("Thickness", worker, on_success, "Error")


def main() -> None:
    if not QT_AVAILABLE:
        raise SystemExit(
            "PySide6 is not available. Install PySide6 to run the GUI."
        )
    app = QApplication.instance() or QApplication(sys.argv)
    gui = ImpedanceGui()
    gui.show()
    app.exec()


if __name__ == "__main__":
    main()
