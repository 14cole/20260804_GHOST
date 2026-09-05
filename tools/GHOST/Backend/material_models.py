"""User-facing material meanings, separate from the legacy numeric file codes."""

MATERIAL_MODELS = {
    1: ("Free sheet: impedance / thin dielectric", ("Air", "Sheet", "Air"),
        "Choose a surface flag: complex sheet impedance, or a thin_dielectric "
        "row with thickness in metres and a dielectric material flag. The thin "
        "layer transmits and includes normal polarization. It is a first-order "
        "2D approximation for a uniform layer in air: electrical thickness <= "
        "0.15 and thickness/curvature radius <= 0.05. Validate against bulk for "
        "your application. Thin layers currently require an all-layer geometry. "
        "Impedance sheets can also couple to pure PEC in 2D. BoR supports "
        "electric impedance sheets alone or joined to PEC along one meridian; "
        "disconnected sheets and thin dielectric layers are not yet supported there."),
    2: ("Opaque conductor: PEC / IBC", ("Air", "Surface", "Conductor"),
        "IBC flag 0 gives PEC. A positive flag specifies Zs = R + jX. "
        "The conductor interior is excluded; this model does not transmit "
        "through the conductor. BoR uses a combined-field formulation for a "
        "uniform reactive IBC on a closed conductor. For a collapsed PEC-backed "
        "FREDDY stack, assign its nominal IBC CSV here in either 2D or BoR. "
        "Draw the outer coating envelope and omit the explicit bulk layers. "
        "The CSV is a normal-incidence scalar approximation; use FREDDY's "
        "coating check to assess planar angle sensitivity."),
    3: ("Bulk material / air", ("Air", "Interface", "Dielectric"),
        "A penetrable isotropic material with complex relative epsilon and "
        "mu. Geometry explicitly describes its boundary. Assign pos_mat; "
        "do not assign an IBC flag on this interface."),
    4: ("Material / conductor", ("Dielectric", "Surface", "Conductor"),
        "The inner boundary of a dielectric coating. IBC flag 0 gives PEC. "
        "A nonzero backing impedance is supported by 2D, but is not currently "
        "supported by BoR."),
    5: ("Material / material", ("pos_mat", "Interface", "neg_mat"),
        "A transmission interface between two distinct isotropic materials. "
        "The normal points from neg_mat into pos_mat. No impedance layer is "
        "implied by this interface."),
}


def segment_type_options():
    return [(str(key), f"{key}: {row[0]}") for key, row in MATERIAL_MODELS.items()]


def choose_thin_layer(parent, dielectric_options):
    """Return a typed surface-material row (without its flag), or None."""
    try:
        from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel
    except ImportError:
        from PySide2.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel
    dialog = QDialog(parent)
    dialog.setWindowTitle("Thin dielectric layer in air (2D)")
    layout = QFormLayout(dialog)
    material = QComboBox()
    for flag, label in dielectric_options:
        material.addItem(label, flag)
    thickness = QDoubleSpinBox()
    thickness.setDecimals(6)
    thickness.setRange(.000001, 1e6)
    thickness.setValue(1.)
    thickness.setSuffix(" mm")
    layout.addRow("Dielectric material", material)
    layout.addRow("Physical thickness", thickness)
    note = QLabel("Draw the layer midsurface as TYPE 1 and assign the new surface flag. "
                  "Use air (0) on both sides. This model retains transmission and phase; "
                  "the solver checks electrical thickness and curvature limits.")
    note.setWordWrap(True)
    layout.addRow(note)
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addRow(buttons)
    if dialog.exec() != QDialog.Accepted:
        return None
    return ["thin_dielectric", format(thickness.value() / 1000., ".12g"), str(material.currentData())]


def show_material_guide(parent):
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QLabel, QVBoxLayout
    except ImportError:
        from PySide2.QtCore import Qt
        from PySide2.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QLabel, QVBoxLayout
    dialog = QDialog(parent)
    dialog.setWindowTitle("Material models and boundary sides")
    dialog.resize(560, 300)
    layout = QVBoxLayout(dialog)
    choice = QComboBox(dialog)
    for key, row in MATERIAL_MODELS.items():
        choice.addItem(f"TYPE {key}: {row[0]}", key)
    layout.addWidget(choice)
    diagram, description = QLabel(), QLabel()
    diagram.setTextFormat(Qt.RichText)
    description.setWordWrap(True)
    layout.addWidget(diagram)
    layout.addWidget(description)
    convention = QLabel("Passive materials use Im(epsilon), Im(mu) <= 0 and Re(Zs) >= 0 under exp(+j omega t).")
    convention.setWordWrap(True)
    layout.addWidget(convention)
    buttons = QDialogButtonBox(QDialogButtonBox.Close)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    def update():
        _, sides, explanation = MATERIAL_MODELS[choice.currentData()]
        diagram.setText(
            '<table width="100%" cellspacing="0" cellpadding="16"><tr>'
            + ''.join(f'<td align="center" bgcolor="{color}"><font color="white">{label}</font></td>'
                      for label, color in zip(sides, ("#475569", "#0e7490", "#475569")))
            + '</tr></table>')
        description.setText(explanation)
    choice.currentIndexChanged.connect(update)
    update()
    dialog.exec()
