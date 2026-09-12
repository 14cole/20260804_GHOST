"""Layer, sheet, and measured-material editors for FREDDY."""
from __future__ import annotations

import math
import copy
from pathlib import Path
from .ui_controls import (
    BooleanVar,
    StringVar,
    bind_check_box,
    bind_line_edit,
    filedialog,
    make_combo,
    messagebox,
)
from .compute import LayerConfig
from .io import read_material_table, constant_material_from_layer, CONSTANT_VALUE_FIELDS
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


def _parse_optional_thickness(text: str, label: str) -> float | None:
    stripped = text.strip()
    if not stripped:
        return None
    value = float(stripped)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and > 0.")
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
        self._initial_tolerances = copy.deepcopy(initial.tolerances) if initial else {}
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
        self.source_var = StringVar('Constant εr / μr (all frequencies)' if init.is_constant else 'Measured material CSV')
        self.constant_vars = {key: StringVar(f'{getattr(init, key):g}') for key in CONSTANT_VALUE_FIELDS}
        default_min = 0.5 * init.thickness_in if initial is None else None
        default_max = 1.5 * init.thickness_in if initial is None else None
        default_step = init.thickness_in / 25.0 if initial is None else None
        self.inv_t_min_var = StringVar(
            f"{(init.inv_t_min_in if init.inv_t_min_in is not None else default_min):g}"
            if init.inv_t_min_in is not None or default_min is not None else ""
        )
        self.inv_t_max_var = StringVar(
            f"{(init.inv_t_max_in if init.inv_t_max_in is not None else default_max):g}"
            if init.inv_t_max_in is not None or default_max is not None else ""
        )
        self.inv_t_acc_var = StringVar(
            f"{(init.inv_t_accuracy_in if init.inv_t_accuracy_in is not None else default_step):g}"
            if init.inv_t_accuracy_in is not None or default_step is not None else ""
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
        file0_edit.setToolTip("Comma-separated .csv with required header: frequency_hz,eps_real,eps_imag,mu_real,mu_imag. Frequency in Hz; relative epsilon and mu.")
        bind_line_edit(self.file_0deg_var, file0_edit)
        grid.addWidget(file0_edit, 3, 1)
        browse0 = QPushButton("Browse")
        browse0.clicked.connect(self._browse_0deg)
        grid.addWidget(browse0, 3, 2)

        self.lbl_90 = QLabel("File (90 deg)")
        grid.addWidget(self.lbl_90, 4, 0, Qt.AlignLeft)
        self.ent_90 = QLineEdit()
        self.ent_90.setToolTip("Comma-separated .csv with required header: frequency_hz,eps_real,eps_imag,mu_real,mu_imag. Frequency in Hz; relative epsilon and mu.")
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
            QLabel("Inverse-design range — choose the allowed thicknesses"),
            7,
            0,
            1,
            3,
            Qt.AlignLeft,
        )
        grid.addWidget(QLabel("Minimum thickness (in)"), 8, 0, Qt.AlignLeft)
        tmin_edit = QLineEdit()
        bind_line_edit(self.inv_t_min_var, tmin_edit)
        grid.addWidget(tmin_edit, 8, 1)
        grid.addWidget(QLabel("Maximum thickness (in)"), 9, 0, Qt.AlignLeft)
        tmax_edit = QLineEdit()
        bind_line_edit(self.inv_t_max_var, tmax_edit)
        grid.addWidget(tmax_edit, 9, 1)
        grid.addWidget(QLabel("Thickness step (in)"), 10, 0, Qt.AlignLeft)
        tacc_edit = QLineEdit()
        bind_line_edit(self.inv_t_acc_var, tacc_edit)
        tacc_edit.setToolTip(
            "Required when varying thickness (for example 0.001 in). Values start "
            "at Minimum and advance by Step without exceeding Maximum."
        )
        grid.addWidget(tacc_edit, 10, 1)

        grid.setColumnStretch(1, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)

        outer = QVBoxLayout(self)
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel('Material source'))
        self.source_combo = make_combo(['Measured material CSV', 'Constant εr / μr (all frequencies)'],
                                       self.source_var, width=310)
        source_row.addWidget(self.source_combo, 1)
        outer.addLayout(source_row)
        self.constant_panel = QWidget()
        constant_layout = QGridLayout(self.constant_panel)
        constant_layout.setContentsMargins(10, 6, 10, 6)
        for index, (key, label) in enumerate(zip(CONSTANT_VALUE_FIELDS, ('εr real', 'εr imaginary', 'μr real', 'μr imaginary'))):
            row, column = divmod(index, 2)
            constant_layout.addWidget(QLabel(label), row, 2 * column)
            entry = QLineEdit()
            bind_line_edit(self.constant_vars[key], entry)
            constant_layout.addWidget(entry, row, 2 * column + 1)
        note = QLabel('Relative εr and μr stay constant at every frequency. Enter signed imaginary parts: negative values represent passive loss in the e^(+jωt) convention.')
        note.setWordWrap(True)
        constant_layout.addWidget(note, 2, 0, 1, 4)
        outer.addWidget(self.constant_panel)
        self.file_widgets = [grid.itemAtPosition(row, col).widget()
                             for row in range(1, 6) for col in range(3)
                             if grid.itemAtPosition(row, col) is not None]
        outer.addLayout(grid)
        outer.addWidget(buttons)
        self.source_var.valueChanged.connect(self._sync_state)
        self._sync_state()

    def _sync_state(self, *_args) -> None:
        constant = self.source_var.get().startswith('Constant')
        self.constant_panel.setVisible(constant)
        for widget in self.file_widgets:
            widget.setVisible(not constant)
        enabled = not constant and self.aniso_var.get()
        for widget in (self.lbl_90, self.ent_90, self.btn_90, self.lbl_pol, self.ent_pol):
            widget.setEnabled(enabled)

    def _browse_0deg(self) -> None:
        p = filedialog.askopenfilename(title="Select 0 deg/isotropic property file", parent=self, filetypes=[("Material CSV (Hz)", "*.csv")])
        if p:
            self.file_0deg_var.set(p)

    def _browse_90deg(self) -> None:
        p = filedialog.askopenfilename(title="Select 90 deg property file", parent=self, filetypes=[("Material CSV (Hz)", "*.csv")])
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
            if not math.isfinite(thickness_in) or thickness_in <= 0:
                raise ValueError("Thickness must be finite and > 0.")
            constant = self.source_var.get().startswith('Constant')
            anisotropic = self.aniso_var.get() and not constant
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

            if not file_0deg and not constant:
                raise ValueError("0 deg/isotropic file is required.")
            if anisotropic and not file_90deg:
                raise ValueError("90 deg file is required for anisotropic layer.")

            inv_t_min_in = _parse_optional_thickness(self.inv_t_min_var.get(), "inv_t_min")
            inv_t_max_in = _parse_optional_thickness(self.inv_t_max_var.get(), "inv_t_max")
            inv_t_accuracy_in = _parse_optional_thickness(
                self.inv_t_acc_var.get(), "thickness step"
            )
            if (inv_t_min_in is None) != (inv_t_max_in is None):
                raise ValueError("Set both minimum and maximum thickness, or leave both blank.")
            if (
                inv_t_min_in is not None
                and inv_t_max_in is not None
                and inv_t_max_in < inv_t_min_in
            ):
                raise ValueError("Maximum thickness must be >= minimum thickness.")

            values = {key: float(value.get()) for key, value in self.constant_vars.items()} if constant else {}
            result = LayerConfig(
                tolerances=copy.deepcopy(self._initial_tolerances),
                thickness_in=thickness_in,
                anisotropic=anisotropic,
                file_0deg='' if constant else file_0deg,
                file_90deg='' if constant else file_90deg,
                polarization_deg=polarization_deg,
                inv_t_min_in=inv_t_min_in,
                inv_t_max_in=inv_t_max_in,
                inv_t_accuracy_in=inv_t_accuracy_in,
                material_source='constant' if constant else 'file',
                **values,
            )
            if constant:
                constant_material_from_layer(result)
            self.result = result
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
        self._initial_tolerances = copy.deepcopy(initial.tolerances) if initial else {}

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
        default_min = 0.5 * init.sheet_resistance if initial is None else None
        default_max = 1.5 * init.sheet_resistance if initial is None else None
        default_step = max(1.0, init.sheet_resistance / 50.0) if initial is None else None
        self.inv_rs_min_var = StringVar(
            f"{(init.inv_rs_min if init.inv_rs_min is not None else default_min):g}"
            if init.inv_rs_min is not None or default_min is not None else ""
        )
        self.inv_rs_max_var = StringVar(
            f"{(init.inv_rs_max if init.inv_rs_max is not None else default_max):g}"
            if init.inv_rs_max is not None or default_max is not None else ""
        )
        self.inv_rs_acc_var = StringVar(
            f"{(init.inv_rs_accuracy if init.inv_rs_accuracy is not None else default_step):g}"
            if init.inv_rs_accuracy is not None or default_step is not None else ""
        )

        grid = QGridLayout()
        grid.setContentsMargins(10, 10, 10, 10)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        grid.addWidget(QLabel("Sheet resistance (\u03a9/sq)"), 0, 0, Qt.AlignLeft)
        rs_edit = QLineEdit()
        bind_line_edit(self.rs_var, rs_edit)
        rs_edit.setToolTip("Nominal resistance; used when no inverse-design range is set.")
        grid.addWidget(rs_edit, 0, 1)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        grid.addWidget(sep, 1, 0, 1, 2)
        grid.addWidget(
            QLabel("Optimization range (leave minimum and maximum blank to keep fixed)"),
            2,
            0,
            1,
            2,
            Qt.AlignLeft,
        )
        grid.addWidget(QLabel("Minimum resistance (\u03a9/sq)"), 3, 0, Qt.AlignLeft)
        rmin_edit = QLineEdit()
        bind_line_edit(self.inv_rs_min_var, rmin_edit)
        grid.addWidget(rmin_edit, 3, 1)
        grid.addWidget(QLabel("Maximum resistance (\u03a9/sq)"), 4, 0, Qt.AlignLeft)
        rmax_edit = QLineEdit()
        bind_line_edit(self.inv_rs_max_var, rmax_edit)
        grid.addWidget(rmax_edit, 4, 1)
        grid.addWidget(QLabel("Resistance step (\u03a9)"), 5, 0, Qt.AlignLeft)
        racc_edit = QLineEdit()
        bind_line_edit(self.inv_rs_acc_var, racc_edit)
        racc_edit.setToolTip(
            "Required when varying resistance (for example 1 ohm). Values start "
            "at Minimum and advance by Step without exceeding Maximum."
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
                self.inv_rs_acc_var.get(), "resistance step"
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
                tolerances=copy.deepcopy(self._initial_tolerances),
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
        file_edit.setToolTip("Comma-separated .csv with required header: frequency_hz,eps_real,eps_imag,mu_real,mu_imag. Frequency in Hz; relative epsilon and mu.")
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
        p = filedialog.askopenfilename(title="Select property file", parent=self, filetypes=[("Material CSV (Hz)", "*.csv")])
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
            read_material_table(Path(file_str))
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
