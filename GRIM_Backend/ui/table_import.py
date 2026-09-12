"""Adapt FREDDY's shared column editor to a GRIM RCS import."""
from __future__ import annotations

from dataclasses import asdict
import importlib

from PySide6.QtWidgets import QComboBox, QLabel

from GRIM_Backend.integrations.freddy import FREDDY_PACKAGE_NAMESPACE, load_freddy_package
from GRIM_Backend.io.mapped_table import MAGNITUDE_FORMATS, grid_from_mapped_rows


def create_table_import_dialog(parent, path):
    # Reuse the authoritative standalone tool through GRIM's private package.
    load_freddy_package()
    ui = importlib.import_module(FREDDY_PACKAGE_NAMESPACE + ".converter_dialog")
    core = importlib.import_module(FREDDY_PACKAGE_NAMESPACE + ".table_conversion")

    class TableImportDialog(ui.FileConverterDialog):
        def __init__(self):
            super().__init__(parent, importing=True)
            self.dataset = None
            self.intro.setText(
                "GRIM could not recognize this table. Label its columns below. "
                "Angles are azimuth/elevation in GRIM's conic convention. "
                "Use constants for missing columns (for example elevation = 0). "
                "Leave Phase unchecked when no phase was measured."
            )
            self.magnitude_format = QComboBox()
            self.magnitude_format.addItems(("Choose magnitude representation…", *MAGNITUDE_FORMATS))
            self.extra_layout.addWidget(QLabel("Magnitude represents"))
            self.extra_layout.addWidget(self.magnitude_format)
            self.path_edit.setText(str(path))
            self.refresh_preview()

        def configure_mapping(self):
            aliases = {
                "frequency": ("frequency", "freq", "f"),
                "azimuth": ("azimuth", "az", "azi"),
                "elevation": ("elevation", "el", "elev"),
                "polarization": ("polarization", "polarisation", "pol"),
                "magnitude": ("magnitude", "mag", "amplitude", "rcs", "power",
                              "magnitude_dbsm", "magnitude_dbke", "magnitude_power_linear"),
                "phase": ("phase", "pha"),
            }
            for name, keys in aliases.items():
                candidates = []
                for i, label in enumerate(self.names):
                    base, unit = core.suggest_column(label)
                    if base.lower() in keys:
                        candidates.append((i, unit))
                source, unit = candidates[0] if len(candidates) == 1 else (None, "As written")
                constant = {"elevation": "0", "azimuth": "0", "polarization": "VV"}.get(name, "")
                input_units = output_units = ("As written",)
                output_unit = "As written"
                if name == "frequency":
                    input_units = ("Choose unit…", "Hz", "kHz", "MHz", "GHz", "THz")
                    if unit not in input_units:
                        unit = "Choose unit…"
                    output_units, output_unit = ("GHz",), "GHz"
                elif name in ("azimuth", "elevation", "phase"):
                    input_units, output_units, output_unit = ("deg", "rad"), ("deg",), "deg"
                    unit = unit if unit in input_units else "deg"
                else:
                    unit = "As written"
                self.add_mapping(name, source=source, constant=constant, input_unit=unit,
                    output_unit=output_unit, checked=name != "phase" or source is not None,
                    locked=True, input_units=input_units, output_units=output_units)

        def submit(self):
            try:
                source, options, columns, names = self.snapshot()
                required = {"frequency", "azimuth", "elevation", "polarization", "magnitude"}
                if not required.issubset(column.name for column in columns):
                    raise ValueError("Frequency, azimuth, elevation, polarization and magnitude are required; "
                                     "use constants for missing columns.")
                representation = self.magnitude_format.currentText()
                if representation not in MAGNITUDE_FORMATS:
                    raise ValueError("Choose what the magnitude column represents.")
                provenance = {"options": asdict(options), "columns": [asdict(c) for c in columns],
                              "source_headers": list(names), "magnitude_format": representation}
                self.start_job(lambda: grid_from_mapped_rows(
                    core.converted_rows(source, options, columns, names), source, representation,
                    mapping=provenance))
            except Exception as exc:
                self.status.setText(str(exc))

        def conversion_finished(self, result):
            self.dataset = result
            self.accept()

    return TableImportDialog()
